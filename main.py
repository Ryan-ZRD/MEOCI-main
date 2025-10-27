import argparse
import torch
from config import Config
from dataset.bdd100k_processor import get_bdd100k_dataloader
from model.alexnet import MultiExitAlexNet
from model.vgg16 import MultiExitVGG16
from adp_d3qn.training import train_adp_d3qn
from adp_d3qn.vehicle_inference import VehicleInference
from adp_d3qn.edge_inference import EdgeInference


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MEOCI: Model Partitioning and Early-exit Joint Optimization (Paper Code)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"], help="Run mode: train/infer")
    parser.add_argument("--model", type=str, default="VGG16", choices=["AlexNet", "VGG16"],
                        help="DNN model (Paper supports AlexNet/VGG16)")
    parser.add_argument("--load_agent", type=str, default=None, help="Path to pre-trained ADP-D3QN agent")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # 1. 配置模型（论文支持AlexNet/VGG16）
    if args.model == "AlexNet":
        config.model_name = "MultiExitAlexNet"
        model = MultiExitAlexNet(num_classes=config.num_classes, ac_min=config.ac_min)
        config.delay_tolerate = 25.0  # AlexNet容忍延迟25ms（论文1-147）
        config.action_dim = 13  # AlexNet12层+1=13个划分点
    else:
        config.model_name = "MultiExitVGG16"
        model = MultiExitVGG16(num_classes=config.num_classes, ac_min=config.ac_min)
        config.delay_tolerate = 250.0  # VGG16容忍延迟250ms（论文1-147）
        config.action_dim = 32  # VGG1631层+1=32个划分点
    model.to(config.device)

    # 2. 数据加载（论文BDD100K数据集）
    print(f"Loading BDD100K Dataset (Paper Experiment)...")
    if args.mode == "train":
        dataloader = get_bdd100k_dataloader(
            data_root=config.data_root,
            split="train",
            batch_size=config.batch_size,
            img_size=config.img_size,
            augment=config.augment
        )
    else:
        dataloader = get_bdd100k_dataloader(
            data_root=config.data_root,
            split="val",
            batch_size=1,
            img_size=config.img_size,
            augment=False
        )

    # 3. 训练模式（ADP-D3QN训练）
    if args.mode == "train":
        print(f"Starting ADP-D3QN Training for {config.model_name} (Paper 4.3节)...")
        agent, metrics = train_adp_d3qn(model=model, dataloader=dataloader, config=config)
        print(f"Training Completed! Best Avg Delay: {metrics['best_avg_delay']:.2f}ms")

    # 4. 推理模式（车边协同推理）
    else:
        if not args.load_agent:
            raise ValueError("Please specify pre-trained agent path (--load_agent) for inference")

        # 加载预训练智能体
        from adp_d3qn.adp_d3qn_agent import ADPD3QNAgent
        agent = ADPD3QNAgent(state_dim=config.state_dim, action_dim=config.action_dim, config=config.__dict__)
        agent.load_model(args.load_agent)
        print(f"Loaded Pre-trained ADP-D3QN Agent: {args.load_agent}")

        # 初始化车端与边缘端推理器
        vehicle_infer = VehicleInference(model=model, config=config.__dict__)
        edge_infer = EdgeInference(model=model, config=config.__dict__)

        # 车边协同推理（论文流程）
        print(f"Starting Vehicle-Edge Collaborative Inference (Paper MEOCI机制)...")
        total_delay = []
        total_accuracy = []

        for batch in dataloader:
            img = batch["image"].numpy()[0]  # 单样本推理
            label = batch["label"].numpy()[0]
            vehicle_id = batch["img_path"][0].split("/")[-1].split(".")[0]

            # 步骤1：车端推理（早退出判断）
            vehicle_result = vehicle_infer.infer(img)
            if vehicle_result["early_exit"]:
                # 早退出：车端直接返回结果
                total_delay.append(vehicle_result["delay"])
                total_accuracy.append(vehicle_result["accuracy"])
                print(
                    f"Vehicle {vehicle_id} | Early Exit {vehicle_result['exit_idx']} | Delay: {vehicle_result['delay']:.2f}ms | Acc: {vehicle_result['accuracy']:.4f}")
                continue

            # 步骤2：车端→边缘端传输中间特征
            intermediate_feat = vehicle_result["intermediate_feat"]
            edge_infer.add_task_to_queue(
                {"feat": intermediate_feat, "vehicle_id": vehicle_id, "timestamp": vehicle_result["timestamp"]})

            # 步骤3：边缘端推理
            edge_result = edge_infer.infer(intermediate_feat, labels=np.array([label]))
            if edge_result["status"] == "completed":
                # 总延迟=车端延迟+传输延迟+边缘延迟（论文公式11）
                transmission_delay = (intermediate_feat.nbytes / (
                            1024 * 1024)) * 8 / config.bandwidth * 1000  # 传输延迟（ms）
                total_infer_delay = vehicle_result["delay"] + transmission_delay + edge_result["delay"]
                total_delay.append(total_infer_delay)
                total_accuracy.append(edge_result["accuracy"])
                print(
                    f"Vehicle {vehicle_id} | Edge Main Exit | Total Delay: {total_infer_delay:.2f}ms | Acc: {edge_result['accuracy']:.4f}")
            else:
                print(f"Vehicle {vehicle_id} | Task Failed (Edge Queue Full)")

            # 步骤4：基于ADP-D3QN更新下一个划分点
            env_state = [edge_result["accuracy"],
                         edge_infer.get_queue_status()["queue_length"] / config.edge_queue_capacity,
                         edge_infer.get_queue_status()["service_rate"] / config.edge_resource,
                         edge_infer.get_queue_status()["task_arrival_rate"] / 0.25]
            optimal_par = agent.select_action(torch.tensor(env_state, dtype=torch.float32))
            vehicle_infer.update_partition_point(optimal_par)
            edge_infer.update_partition_point(optimal_par)

            # 仅推理10个样本（简化）
            if len(total_delay) >= 10:
                break

        # 打印推理统计（论文5.4节指标）
        avg_delay = np.mean(total_delay)
        avg_accuracy = np.mean(total_accuracy)
        task_completion_rate = len([d for d in total_delay if d <= config.delay_tolerate]) / len(total_delay)
        print(f"\nInference Statistics (Paper Metrics):")
        print(f"Average Inference Delay: {avg_delay:.2f}ms (Tolerate: {config.delay_tolerate}ms)")
        print(f"Average Inference Accuracy: {avg_accuracy:.4f} (Threshold: {config.ac_min})")
        print(f"Task Completion Rate: {task_completion_rate:.4f}")


if __name__ == "__main__":
    main()