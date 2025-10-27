import argparse
import torch
import numpy as np
import time

from config import Config
from dataset.bdd100k_processor import get_bdd100k_dataloader
from adp_d3qn.training import train_adp_d3qn
from adp_d3qn.vehicle_inference import VehicleInference
from adp_d3qn.edge_inference import EdgeInference
from adp_d3qn.adp_d3qn_agent import ADPD3QNAgent
from adp_d3qn.metrics import calculate_avg_delay, calculate_task_completion_rate


def parse_args():
    parser = argparse.ArgumentParser(description="MEOCI (Paper Implementation)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"],
                        help="Run mode: train / infer")
    parser.add_argument("--model", type=str, default="VGG16",
                        choices=["AlexNet", "VGG16", "ResNet50", "YOLOv10"],
                        help="Select DNN backbone for experiment")
    parser.add_argument("--load_agent", type=str, default=None,
                        help="Path to pre-trained ADP-D3QN agent (for inference mode)")
    return parser.parse_args()


def build_model(config):
    """根据论文模型类型加载结构"""
    if config.model_name == "MultiExitAlexNet":
        from model.alexnet import MultiExitAlexNet as Model
    elif config.model_name == "MultiExitVGG16":
        from model.vgg16 import MultiExitVGG16 as Model
    elif config.model_name == "MultiExitResNet50":
        from model.resnet50 import MultiExitResNet50 as Model
    elif config.model_name == "MultiExitYOLOv10":
        from model.yolov10 import MultiExitYOLOv10 as Model
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    model = Model(num_classes=config.num_classes, ac_min=config.ac_min)
    model.to(config.edge_device)
    print(f"[Model Loaded] {config.model_name} | Layers: {config.num_layers} | Exits: {config.exit_count}")
    return model


def main():
    args = parse_args()
    config = Config(model_name=f"MultiExit{args.model}")

    # ----------------------------------------------------------
    # 1. 加载模型与数据（论文 Section 5.1）
    # ----------------------------------------------------------
    model = build_model(config)
    dataloader = get_bdd100k_dataloader(
        config.__dict__,
        split="train" if args.mode == "train" else "val"
    )

    # ----------------------------------------------------------
    # 2. 训练模式（Section 4.3 ADP-D3QN）
    # ----------------------------------------------------------
    if args.mode == "train":
        print(f"\n[Training Mode] ADP-D3QN Optimization for {config.model_name}")
        agent, metrics = train_adp_d3qn(model=model, dataloader=dataloader, config=config)
        print(f"Training Completed ✅ | Best Avg Delay: {metrics['best_avg_delay']:.2f} ms\n")
        return

    # ----------------------------------------------------------
    # 3. 推理模式（Section 5.4 Inference Evaluation）
    # ----------------------------------------------------------
    if not args.load_agent:
        raise ValueError("Inference mode requires --load_agent path")

    print(f"\n[Inference Mode] Vehicle–Edge Collaborative Execution | Model: {config.model_name}")
    agent = ADPD3QNAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        config=config.__dict__
    )
    agent.load_model(args.load_agent)
    print(f"Loaded ADP-D3QN Agent from {args.load_agent}")

    # 初始化推理组件
    vehicle_infer = VehicleInference(model=model, config=config.__dict__)
    edge_infer = EdgeInference(model=model, config=config.__dict__)

    # ----------------------------------------------------------
    # 4. 推理循环（Paper Fig. 7–8 Workflow）
    # ----------------------------------------------------------
    total_delay, total_accuracy = [], []
    for i, batch in enumerate(dataloader):
        try:
            img = batch["image"][0].numpy()
            label = batch["label"][0]
            vid = batch["path"][0].split("/")[-1]

            # Step 1: 车端推理（Eq.6–7）
            vehicle_result = vehicle_infer.infer(img)
            if vehicle_result["early_exit"]:
                total_delay.append(vehicle_result["delay"])
                total_accuracy.append(vehicle_result["accuracy"])
                print(f"[Vehicle {vid}] Early Exit {vehicle_result['exit_idx']} | "
                      f"Delay: {vehicle_result['delay']:.2f} ms | Acc: {vehicle_result['accuracy']:.4f}")
                continue

            # Step 2: 传输 & 加入边缘队列（Eq.8）
            intermediate_feat = vehicle_result["intermediate_feat"]
            edge_infer.add_task_to_queue({"feat": intermediate_feat, "vehicle_id": vid, "timestamp": time.time()})

            # Step 3: 边缘端推理（Eq.8–11）
            edge_result = edge_infer.infer(intermediate_feat, labels=torch.tensor([label]))
            if edge_result["status"] == "completed":
                transmission_delay = (intermediate_feat.nbytes / (1024 * 1024)) * 8 / config.bandwidth * 1000
                total_infer_delay = vehicle_result["delay"] + transmission_delay + edge_result["delay"]
                total_delay.append(total_infer_delay)
                total_accuracy.append(edge_result["accuracy"])
                print(f"[Vehicle {vid}] Edge Main Exit | Delay: {total_infer_delay:.2f} ms | "
                      f"Acc: {edge_result['accuracy']:.4f}")
            else:
                print(f"[Vehicle {vid}] Task dropped (Edge queue full)")

            # Step 4: ADP-D3QN 更新划分点（Eq.15）
            q_status = edge_infer.get_queue_status()
            env_state = [
                edge_result["accuracy"],
                q_status["queue_length"] / config.edge_queue_capacity,
                q_status["service_rate"] / config.edge_resource,
                q_status["task_arrival_rate"] / 0.25
            ]
            optimal_par = agent.select_action(torch.tensor(env_state, dtype=torch.float32))
            vehicle_infer.update_partition_point(optimal_par)
            edge_infer.update_partition_point(optimal_par)

            if len(total_delay) >= 10:
                break

        except Exception as e:
            print(f"[Warning] Skipped sample {i} due to error: {e}")
            continue

    # ----------------------------------------------------------
    # 5. 输出实验指标（Section 5.4 Metrics）
    # ----------------------------------------------------------
    avg_delay = np.mean(total_delay)
    avg_accuracy = np.mean(total_accuracy)
    completion_rate = calculate_task_completion_rate(avg_delay, config.delay_tolerate)

    print("\n================== Inference Evaluation (Paper §5.4) ==================")
    print(f"Average Delay: {avg_delay:.2f} ms (Tolerate: {config.delay_tolerate} ms)")
    print(f"Inference Accuracy: {avg_accuracy:.4f} (Threshold: {config.ac_min})")
    print(f"Task Completion Rate: {completion_rate:.2%}")
    print("=======================================================================")


if __name__ == "__main__":
    main()
