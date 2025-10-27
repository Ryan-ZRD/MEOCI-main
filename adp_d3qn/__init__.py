"""
ADP-D3QN算法模块（论文《MEOCI: Model Partitioning and Early-exit point Selection Joint Optimization for Collaborative Inference in Vehicular Edge Computing》核心优化算法）
包含车边协同环境建模、智能体训练、模型划分与推理等功能
"""
from .adp_d3qn_agent import ADPD3QNAgent
from .env import VECEnv
from .early_exit import calculate_early_exit_probability, validate_early_exit_accuracy
from .model_partition import decide_optimal_partition_point, get_model_layer_compute_cost
from .training import train_adp_d3qn
from .vehicle_inference import VehicleInference
from .edge_inference import EdgeInference
from .metrics import (
    calculate_avg_delay, calculate_task_completion_rate, calculate_inference_accuracy,
    calculate_accuracy_loss, log_experiment_metrics
)

__all__ = [
    "ADPD3QNAgent", "VECEnv", "calculate_early_exit_probability",
    "validate_early_exit_accuracy", "decide_optimal_partition_point",
    "get_model_layer_compute_cost", "train_adp_d3qn", "VehicleInference",
    "EdgeInference", "calculate_avg_delay", "calculate_task_completion_rate",
    "calculate_inference_accuracy", "calculate_accuracy_loss", "log_experiment_metrics"
]