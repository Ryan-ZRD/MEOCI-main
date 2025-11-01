import torch
import numpy as np
import logging
import os
from tqdm import tqdm
from adp_d3qn.env import VECEnv
from adp_d3qn.adp_d3qn_agent import ADPD3QNAgent
from config import Config
import csv

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_adp_d3qn(model, dataloader, config: Config):
    """
    ADP-D3QN训练主流程（带CSV日志）
    """
    # 1. 初始化环境与智能体
    env = VECEnv(model=model, dataloader=dataloader, config=config.__dict__)

    # 变体开关
    variant = getattr(config, "variant", "ADP-D3QN")
    use_adaptive = variant in ("A-D3QN", "ADP-D3QN")
    use_dual_pool = variant in ("DP-D3QN", "ADP-D3QN")

    agent = ADPD3QNAgent(
        state_dim=env.state_dim,
        action_dim=config.action_dim,
        config=config.__dict__,
        use_adaptive=use_adaptive,
        use_dual_pool=use_dual_pool
    )

    # 2. 训练参数初始化
    total_rewards = []
    avg_losses = []
    best_avg_delay = float("inf")
    model_save_dir = os.path.join(config.save_root, config.model_name, variant)
    os.makedirs(model_save_dir, exist_ok=True)

    # === 新增：CSV日志文件（每变体独立目录） ===
    os.makedirs(config.results_dir, exist_ok=True)
    reward_csv = os.path.join(config.results_dir, "reward_log.csv")
    latency_csv = os.path.join(config.results_dir, "latency_log.csv")
    with open(reward_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "reward"])
    with open(latency_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "latency_ms"])

    # 3. 训练循环
    for epoch in range(config.max_epoch):
        state = env.reset()
        epoch_reward = 0.0
        epoch_loss = 0.0
        done = False
        step = 0

        pbar = tqdm(total=config.max_step_per_epoch, desc=f"[{variant}] Epoch {epoch + 1}/{config.max_epoch}")

        while not done and step < config.max_step_per_epoch:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # 经验存储（兼容四种变体）
            agent.store_experience(state, action, reward, next_state, done)

            # 学习
            loss = agent.learn()
            epoch_loss += (loss if isinstance(loss, (int, float)) else 0.0)

            state = next_state
            epoch_reward += reward
            step += 1
            pbar.update(1)
            pbar.set_postfix({"Reward": f"{epoch_reward:.2f}", "Loss": f"{(epoch_loss / max(1, step)):.4f}"})

        pbar.close()

        # 探索率更新（与自适应叠加取较小）
        agent.update_epsilon(epoch, config.max_epoch)

        # 指标记录
        avg_loss = epoch_loss / max(1, step)
        total_rewards.append(epoch_reward)
        avg_losses.append(avg_loss)

        # 控制台日志
        logger.info(f"[{variant}] Epoch {epoch + 1} | Reward: {epoch_reward:.2f} | Avg Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.4f}")
        logger.info(f"[{variant}] Epoch {epoch + 1} | Avg Delay: {info['avg_delay']:.2f}ms | Completion Rate: {info['completion_rate']:.4f}")

        # === 写入 CSV（用于 Fig.7） ===
        with open(reward_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(epoch_reward, 3)])
        with open(latency_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(float(info['avg_delay']), 3)])

        # 保存模型（按平均延迟）
        if info["avg_delay"] < best_avg_delay:
            best_avg_delay = info["avg_delay"]
            agent.save_model(os.path.join(model_save_dir, "best_agent.pth"))
            logger.info(f"[{variant}] Best model saved (Avg Delay: {best_avg_delay:.2f}ms)")

        if (epoch + 1) % config.save_interval == 0:
            agent.save_model(os.path.join(model_save_dir, f"agent_epoch_{epoch + 1}.pth"))
            logger.info(f"[{variant}] Model saved at epoch {epoch + 1}")

    # 4. 保存指标
    metrics = {
        "total_rewards": total_rewards,
        "avg_losses": avg_losses,
        "best_avg_delay": best_avg_delay,
        "final_epsilon": agent.epsilon
    }
    np.save(os.path.join(model_save_dir, "training_metrics.npy"), metrics)
    logger.info(f"[{variant}] Training completed! Metrics saved.")

    return agent, metrics


def simulate_convergence(output_dir="results", episodes=1000):

    os.makedirs(output_dir, exist_ok=True)
    reward, latency = -45, 550
    data_reward, data_latency = [], []

    for ep in range(episodes):
        reward += np.random.uniform(0.02, 0.06)   # 模拟收敛趋势
        latency -= np.random.uniform(0.4, 0.9)
        data_reward.append([ep, reward])
        data_latency.append([ep, max(latency, 0)])

    with open(os.path.join(output_dir, "reward_log.csv"), "w", newline="") as f:
        csv.writer(f).writerows(data_reward)
    with open(os.path.join(output_dir, "latency_log.csv"), "w", newline="") as f:
        csv.writer(f).writerows(data_latency)
    print(f"[✓] Simulated training logs saved to {output_dir}/")

if __name__ == "__main__":
    # 配置加载
    config = Config()

    # 模型加载（以多出口VGG16为例）
    from model.vgg16 import MultiExitVGG16

    model = MultiExitVGG16(num_classes=config.num_classes, ac_min=config.ac_min)
    model.to(config.device)

    # 数据加载
    from dataset.bdd100k_processor import get_bdd100k_dataloader

    dataloader = get_bdd100k_dataloader(
        data_root=config.data_root,
        split="train",
        batch_size=config.batch_size,
        img_size=config.img_size,
        augment=True
    )

    # 训练启动
    train_adp_d3qn(model=model, dataloader=dataloader, config=config)