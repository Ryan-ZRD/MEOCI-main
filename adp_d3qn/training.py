import torch
import numpy as np
import logging
import os
from tqdm import tqdm
from adp_d3qn.env import VECEnv
from adp_d3qn.adp_d3qn_agent import ADPD3QNAgent
from config import Config

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_adp_d3qn(model, dataloader, config: Config):
    """
    ADP-D3QN训练主流程
    :param model: 多出口DNN模型
    :param dataloader: 数据加载器
    :param config: 配置类实例
    """
    # 1. 初始化环境与智能体
    env = VECEnv(model=model, dataloader=dataloader, config=config.__dict__)
    agent = ADPD3QNAgent(
        state_dim=env.state_dim,
        action_dim=config.action_dim,
        config=config.__dict__
    )

    # 2. 训练参数初始化
    total_rewards = []
    avg_losses = []
    best_avg_delay = float("inf")
    model_save_dir = os.path.join(config.save_root, config.model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # 3. 训练循环
    for epoch in range(config.max_epoch):
        state = env.reset()
        epoch_reward = 0.0
        epoch_loss = 0.0
        done = False
        step = 0

        # 进度条
        pbar = tqdm(total=config.max_step_per_epoch, desc=f"Epoch {epoch + 1}/{config.max_epoch}")

        while not done and step < config.max_step_per_epoch:
            # 3.1 动作选择
            action = agent.select_action(state)

            # 3.2 环境交互
            next_state, reward, done, info = env.step(action)

            # 3.3 经验存储
            agent.store_experience(state, action, reward, next_state, done)

            # 3.4 智能体学习（经验池满后开始）
            if len(agent.E1) + len(agent.E2) >= config.batch_size:
                loss = agent.learn()
                epoch_loss += loss

            # 3.5 状态更新
            state = next_state
            epoch_reward += reward
            step += 1
            pbar.update(1)
            pbar.set_postfix({"Reward": f"{epoch_reward:.2f}", "Loss": f"{epoch_loss / step:.4f}" if step > 0 else 0})

        pbar.close()

        # 3.6 探索率更新
        agent.update_epsilon(epoch, config.max_epoch)

        # 3.7 指标记录
        avg_loss = epoch_loss / step if step > 0 else 0
        total_rewards.append(epoch_reward)
        avg_losses.append(avg_loss)

        # 3.8 日志输出
        logger.info(
            f"Epoch {epoch + 1} | Reward: {epoch_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")
        logger.info(
            f"Epoch {epoch + 1} | Avg Delay: {info['avg_delay']:.2f}ms | Completion Rate: {info['completion_rate']:.4f}")

        # 3.9 模型保存（基于平均延迟）
        if info["avg_delay"] < best_avg_delay:
            best_avg_delay = info["avg_delay"]
            agent.save_model(os.path.join(model_save_dir, "best_agent.pth"))
            logger.info(f"Best model saved (Avg Delay: {best_avg_delay:.2f}ms)")

        # 3.10 定期保存模型
        if (epoch + 1) % config.save_interval == 0:
            agent.save_model(os.path.join(model_save_dir, f"agent_epoch_{epoch + 1}.pth"))
            logger.info(f"Model saved at epoch {epoch + 1}")

    # 4. 训练结束：保存指标
    metrics = {
        "total_rewards": total_rewards,
        "avg_losses": avg_losses,
        "best_avg_delay": best_avg_delay,
        "final_epsilon": agent.epsilon
    }
    np.save(os.path.join(model_save_dir, "training_metrics.npy"), metrics)
    logger.info("Training completed! Metrics saved.")

    return agent, metrics


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