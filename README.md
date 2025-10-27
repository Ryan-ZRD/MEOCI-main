# MEOCI: Model Partitioning and Early-Exit Point Selection Joint Optimization for Collaborative Inference in Vehicular Edge Computing

Official implementation of the paper  
> **"MEOCI: Model Partitioning and Early-Exit Point Selection Joint Optimization for Collaborative Inference in Vehicular Edge Computing"**  


This repository provides the **complete implementation and experimental framework** of the MEOCI algorithm — a **model partitioning and early-exit joint optimization mechanism** for **edge–vehicle collaborative inference acceleration** using the **Adaptive Dual-Pool Dueling Double DQN (ADP-D3QN)** algorithm.

---

## 🧭 Table of Contents
- [Introduction](#introduction)
- [Framework Overview](#framework-overview)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Training the ADP-D3QN Agent](#training-the-adp-d3qn-agent)
  - [Collaborative Inference](#collaborative-inference)
  - [Evaluation and Visualization](#evaluation-and-visualization)
- [Models and Datasets](#models-and-datasets)
- [Algorithm Details](#algorithm-details)
- [Experimental Results](#experimental-results)
- [Citation](#citation)

---

## 🚀 Introduction

### 🔍 Background
In **Vehicular Edge Computing (VEC)**, Deep Neural Networks (DNNs) are the backbone of intelligent driving applications.  
However, when multiple vehicles offload inference tasks simultaneously, **edge servers experience computational overload and excessive latency**, threatening the safety and real-time requirements of autonomous systems.

### 🧩 Motivation
To address these challenges, **MEOCI** integrates **model partitioning** and **multi-exit early-exit mechanisms**, aiming to:
- Dynamically decide **where to split DNN layers** between vehicle and RSU (partitioning);
- Select the **optimal early-exit point** based on accuracy thresholds;
- Minimize **average inference latency** under **accuracy constraints**.

---

## 🧠 Framework Overview

The MEOCI framework establishes **collaborative inference** between **vehicles** and **edge RSUs** (Fig. 1 in paper):

1. Vehicles generate DNN tasks and send requests to RSUs.  
2. The ADP-D3QN agent determines **partitioning and early-exit points**.  
3. Vehicles perform local shallow-layer inference; RSUs handle deeper layers.  
4. Results are aggregated and sent back to vehicles.

This framework dynamically adapts to **real-time network conditions, computing load, and task complexity**, ensuring reliable low-latency inference.

---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.8  
- PyTorch ≥ 2.1.0  
- CUDA ≥ 11.8 (for GPU acceleration)  
- Dataset: [BDD100K](https://bdd-data.berkeley.edu/)  
- Additional dependencies listed in `requirements.txt`

### Setup
```bash
git clone https://github.com/YourUsername/meoci.git
cd meoci
pip install -r requirements.txt
````

---

## 📁 Directory Structure

```
meoci/
├── adp_d3qn/                 # ADP-D3QN optimization algorithm
│   ├── adp_d3qn_agent.py     # Core agent (dual experience pool + ε-greedy)
│   ├── env.py                # Vehicular Edge Computing (VEC) environment
│   ├── model_partition.py    # Layer-wise DNN partition logic
│   ├── early_exit.py         # Multi-exit decision module
│   ├── training.py           # DRL training process
│   ├── vehicle_inference.py  # Local inference (vehicle-side)
│   ├── edge_inference.py     # Edge inference (RSU-side)
│   ├── metrics.py            # Latency, accuracy, energy metrics
│   └── __init__.py
├── model/
│   ├── base_model.py         # Unified multi-exit base class
│   ├── alexnet.py            # Multi-exit AlexNet (4 exits)
│   ├── vgg16.py              # Multi-exit VGG16 (5 exits)
│   ├── resnet50.py           # Multi-exit ResNet50 (6 exits)
│   └── yolov10.py            # Multi-exit YOLOv10 (3 exits, detection)
├── dataset/
│   ├── bdd100k_processor.py  # Data loading and preprocessing
│   ├── data_augmentation.py  # Data augmentation utilities
│   └── dataset_utils.py
├── config.py                 # Hyperparameters and experiment settings
├── main.py                   # Entry for training/inference/evaluation
├── requirements.txt          # Dependency list
└── README.md                 # Documentation
```

---

## 💻 Usage

### 🏋️ Training the ADP-D3QN Agent

Train the agent to jointly learn **partitioning** and **early-exit decisions**:

```bash
python main.py --mode train --model vgg16 --dataset bdd100k
```

### ⚡ Collaborative Inference

Perform inference using the trained agent:

```bash
python main.py --mode infer --model vgg16 --agent_path saved_models/best_agent.pth
```

### 📈 Evaluation and Visualization

Reproduce and visualize experimental metrics:

```bash
python main.py --mode evaluate --model alexnet --agent_path saved_models/best_agent.pth
```

---

## 🧩 Models and Datasets

### Supported Multi-Exit Models

| Model                 | Exit Points | Description                               |
| --------------------- | ----------- | ----------------------------------------- |
| **MultiExitAlexNet**  | 4           | Lightweight CNN for simple classification |
| **MultiExitVGG16**    | 5           | Deep CNN with high accuracy               |
| **MultiExitResNet50** | 6           | Residual network with enhanced stability  |
| **MultiExitYOLOv10**  | 3           | Real-time object detection network        |

Each exit point satisfies an **accuracy constraint** `a_c_i(t) > a_c_min (0.8)` to ensure early exits maintain acceptable precision.

### Dataset

Experiments are conducted using **BDD100K**, resized and preprocessed for efficient edge inference simulation.

---

## 🧮 Algorithm Details

### 🔹 Problem Definition

Jointly optimize:
[
\min_{par(t), exit(t)} ; \text{delay}*{avg} \quad
s.t. ; a_c(t) > a_c*{min}, ; con_i(t) + con_e(t) \le con_{tol}
]
where `par(t)` is the partition point, `exit(t)` is the early-exit index, and the objective is minimizing **average inference latency** under accuracy and energy constraints.

### 🔹 Markov Decision Process (MDP)

* **State**: `(accuracy, queue length, edge resource, task rate)`
* **Action**: `(partition point, early-exit point)`
* **Reward**: `R(t) = - delay_avg(t)`
* **Policy**: Improved ε–greedy with dual replay pools `(E1, E2)`

### 🔹 ADP-D3QN Innovations

1. **Adaptive ε–greedy exploration**: dynamically adjusts exploration ratio with training epochs.
2. **Dual Experience Pool**: balances exploitation of high-Q samples and exploration of low-Q ones.
3. **Improved Convergence**: 15–30% faster stabilization compared to vanilla D3QN.

---

## 🧪 Experimental Results

### ⚙️ Experimental Setup

* **Edge Platform:** K3S + Docker Cluster (1 Master + 3 Nodes)
* **Vehicle Nodes:** Raspberry Pi 4B and Jetson Nano
* **Network Control:** COMCAST bandwidth simulator
* **Framework:** PyTorch + CUDA + Rancher visualization

### 📊 Baseline Algorithms

* **Vehicle-Only:** Local-only inference
* **Edge-Only:** Full offloading
* **Neur:** Layer-level partition without early-exit
* **Edgent:** Single-exit adaptive offloading
* **DINA (Fog-based):** Distributed inference via fog nodes
* **FedAdapt:** Federated adaptive split learning
* **LBO:** DRL-based online DNN partitioning and exit decision

### 🧷 Key Metrics

* **Average Inference Latency (ms)**
* **Task Completion Rate (%)**
* **Inference Accuracy (%)**
* **Energy Consumption (J)**
* **Early-Exit Probability Distribution**

### 🧩 Highlights

* ADP-D3QN reduces **average latency by 15.8% (AlexNet)** and **8.7% (VGG16)** over Edgent.
* Under high-load (25 vehicles), MEOCI maintains **>90% completion rate** while minimizing queuing delay.
* The multi-exit models reduce redundant computation with <1.2% accuracy loss.
* Applicable to **heterogeneous hardware (Pi 4B, Jetson Nano)** with scalable acceleration performance.

---

[//]: # (## 📚 Citation)

[//]: # ()
[//]: # (If you find this work useful, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{li2025meoci,)

[//]: # (  title   = {MEOCI: Model Partitioning and Early-Exit Point Selection Joint Optimization for Collaborative Inference in Vehicular Edge Computing},)

[//]: # (  author  = {Chunlin Li and Jiaqi Wang and Kun Jiang and Cheng Xiong and Shaohua Wan},)

[//]: # (  journal = {IEEE Transactions on Intelligent Transportation Systems},)

[//]: # (  year    = {2025},)

[//]: # (  volume  = {XX},)

[//]: # (  number  = {XX},)

[//]: # (  pages   = {XXXX--XXXX},)

[//]: # (  doi     = {10.1109/TITS.2025.XXXXXXX})

[//]: # (})

[//]: # (```)

---

## ✅ Reproducibility Notes

* All configurations (bandwidth, power, delay constraints) are defined in `config.py`.
* Trained agents and pre-trained model weights will be available under `saved_models/`.
* Simulation logs and figures correspond to **Fig. 7–10** of the paper.
* Real hardware results were obtained using Raspberry Pi 4B and Jetson Nano devices.

---

© 2025 Wuhan University of Technology & University of Electronic Science and Technology of China.
All rights reserved.


---

## 📈 Results Visualization

To help reproduce and visualize the experimental results presented in **Figures 7 – 10** of the MEOCI paper, we provide plotting utilities under the `visualization/` directory.  
These scripts generate key figures such as **training convergence**, **early-exit distribution**, and **latency/performance curves** for comparison with baseline algorithms.

### 📊 Directory
```

visualization/
├── plot_convergence.py          # Fig. 7: Convergence comparison (D3QN vs ADP-D3QN)
├── plot_exit_distribution.py    # Fig. 8: Exit probability & accuracy for AlexNet / VGG16
├── plot_vehicle_latency.py      # Fig. 9–10: Latency vs vehicle count / device type
├── plot_completion_rate.py      # Task completion rate under varying loads
└── utils.py                     # Common plotting utilities

````

### 🧩 Convergence Analysis (Fig. 7)

```bash
python visualization/plot_convergence.py --input results/reward_log.csv
```

📊 Visualization Result:

<p align="center">
  <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/P1.png" alt="Fig7" width="45%"/>
</p>

<p align="center">
  <b>Fig. 7.</b> Comparison of convergence between different algorithmic and environmental settings.
</p>

Description:

Fig. 7 illustrates the convergence comparison among different reinforcement learning algorithms under identical vehicular edge computing environments.
The proposed ADP-D3QN (Adaptive Dual-Pool Dueling Double Deep Q-Network) achieves the highest reward level and fastest convergence speed, indicating stronger learning stability and policy optimization capability.
Compared to D3QN, A-D3QN, and DP-D3QN, the proposed approach exhibits smoother learning trajectories, reduced reward oscillations after approximately 400 episodes, and stable convergence after 600 episodes.
These results verify that the adaptive dual-pool exploration mechanism in ADP-D3QN effectively balances exploration and exploitation, accelerates convergence, and mitigates instability caused by dynamic vehicular network conditions.

```python
# visualization/plot_convergence.py
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/reward_log.csv")

plt.figure(figsize=(6,4))
for col in df.columns[1:]:
    plt.plot(df["Episode"], df[col], label=col, linewidth=1.8)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Convergence of Different Algorithms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/fig7_convergence.png", dpi=300)
plt.show()
```

### 🧩 Early-Exit Probability and Accuracy Analysis (Fig. 8)

```bash
python visualization/plot_exit_distribution.py --model alexnet
python visualization/plot_exit_distribution.py --model vgg16

```

📊 Visualization Result:

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alexprobability.png" alt="(a) AlexNet" width="95%"/><br>
      <b>(a) AlexNet</b>
    </td>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vggprobability.png" alt="(b) VGG16" width="95%"/><br>
      <b>(b) VGG16</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 8.</b> The accuracy and probability of early exit of multi-exit DNN models.
</p>


Description:

Fig. 8 illustrates the relationship between early-exit probability and classification accuracy across multiple exit branches in the multi-exit DNNs (AlexNet and VGG16).
Each exit corresponds to a potential early-termination point for inference.
As the exit depth increases (from exit 1 to exit 4/5), both exit accuracy and exit activation probability increase.
This indicates that deeper exits capture richer semantic features, allowing higher prediction confidence.
For AlexNet, over 75 % of tasks exit before the final layer, while VGG16 exhibits a more balanced distribution, with exit 5 achieving ≈ 87 % accuracy.
These results verify that the multi-exit structure effectively balances latency and accuracy, enabling adaptive early inference under real-time constraints.

```python
# visualization/plot_exit_distribution.py
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="alexnet or vgg16")
args = parser.parse_args()

if args.model.lower() == "alexnet":
    exits = ["exit1","exit2","exit3","exit4"]
    prob = [68.3, 72.4, 76.8, 83.5]
    acc  = [65.1, 70.3, 73.5, 78.8]
    title = "Early-Exit Probability and Accuracy (AlexNet)"
    save_path = "results/fig8a_exit_alexnet.png"

elif args.model.lower() == "vgg16":
    exits = ["exit1","exit2","exit3","exit4","exit5"]
    prob = [70.5, 74.8, 78.2, 84.1, 89.0]
    acc  = [67.2, 72.9, 75.6, 80.4, 86.9]
    title = "Early-Exit Probability and Accuracy (VGG16)"
    save_path = "results/fig8b_exit_vgg16.png"

else:
    raise ValueError("Unsupported model type.")

fig, ax1 = plt.subplots(figsize=(6,4))
ax1.bar(exits, prob, color="royalblue", label="Exit Probability (%)")
ax2 = ax1.twinx()
ax2.plot(exits, acc, "orange", marker="o", linewidth=2, label="Accuracy (%)")

ax1.set_xlabel("Exit Point")
ax1.set_ylabel("Exit Probability (%)")
ax2.set_ylabel("Accuracy (%)")
plt.title(title)
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
```

### 🧩 Performance of Heterogeneous Vehicles (Fig. 9)

```bash
python visualization/plot_heterogeneous_performance.py --model alexnet --data results/device_latency_alexnet.csv
python visualization/plot_heterogeneous_performance.py --model vgg16 --data results/device_latency_vgg16.csv
```
📊 Visualization Result:

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alexnetdevice.png" alt="(a) AlexNet" width="95%"/><br>
      <b>(a) AlexNet</b>
    </td>
    <td align="center" width="50%">
      <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/vgg16device.png" alt="(b) VGG16" width="95%"/><br>
      <b>(b) VGG16</b>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig. 9.</b> Performance of heterogeneous vehicles in multi-exit DNN models.
</p>

Description:

Fig. 9 compares the inference latency of heterogeneous devices (Jetson Nano and Raspberry Pi 4B) executing multi-exit DNN models (AlexNet and VGG16) under different collaborative inference strategies.
The proposed ADP-D3QN consistently achieves the lowest inference latency across both devices, outperforming baselines such as Vehicle-Only, Edge-Only, Neur, Edgent, FedAdapt, and DINA (Fog-Based).
On the Raspberry Pi 4B, MEOCI reduces latency by 25 % – 35 % compared with FedAdapt and DINA, while maintaining accuracy comparable to larger models.
These findings confirm that MEOCI effectively.

```python
# visualization/plot_heterogeneous_performance.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="alexnet or vgg16")
parser.add_argument("--data", type=str, required=True, help="path to csv file")
args = parser.parse_args()

df = pd.read_csv(args.data)
models = ["Vehicle-Only","Edge-Only","Neur","Edgent","DINA(Fog-Based)","FedAdapt","LBO","ADP-D3QN"]

nano = df.loc[df["Device"]=="Nano", models].values.flatten()
pi4b = df.loc[df["Device"]=="Pi4B", models].values.flatten()

x = range(len(models))
bar_width = 0.35
plt.figure(figsize=(8,5))
plt.bar(x, nano, bar_width, label="Nano")
plt.bar([i + bar_width for i in x], pi4b, bar_width, label="Pi 4B")
plt.xticks([i + bar_width / 2 for i in x], models, rotation=30)
plt.ylabel("Inference Latency (ms)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

if args.model.lower() == "alexnet":
    plt.title("Heterogeneous Device Performance (AlexNet)")
    plt.savefig("results/fig9a_heter_alexnet.png", dpi=300)
elif args.model.lower() == "vgg16":
    plt.title("Heterogeneous Device Performance (VGG16)")
    plt.savefig("results/fig9b_heter_vgg16.png", dpi=300)
else:
    raise ValueError("Unsupported model type. Choose alexnet or vgg16.")

plt.tight_layout()
plt.show()
```

### 🧩 Effect of Number of Vehicles (Fig. 10)

```bash
python visualization/plot_vehicle_effect.py --metric latency --data results/latency_vs_vehicle_alexnet.csv
python visualization/plot_vehicle_effect.py --metric completion --data results/completion_vs_vehicle_alexnet.csv
```

📊 Visualization Result:

<table align="center"> <tr> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-vehicle-delay-3.png" alt="(a) Inference latency in AlexNet" width="95%"/><br> <b>(a) Inference latency in AlexNet</b> </td> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-numberVehicle-taskCompletion.png" alt="(b) Completion rate in AlexNet" width="95%"/><br> <b>(b) Completion rate in AlexNet</b> </td> </tr> </table> <p align="center"> <b>Fig. 10.</b> Effect of number of vehicles on inference latency and task completion rate. </p>

Description:

Fig. 10 demonstrates the impact of increasing vehicle density on system inference latency and task completion rate under different collaborative inference strategies.
As the number of vehicles increases, both computation congestion and communication delay rise, causing a steady increase in inference latency across all methods.
However, the proposed ADP-D3QN (MEOCI) exhibits the lowest latency growth and maintains the highest completion rate, outperforming conventional schemes such as Vehicle-Only, Edge-Only, and Edgent.
This improvement stems from MEOCI’s ability to dynamically allocate computing resources and select optimal model partition points based on current vehicular load.
In scenarios with 25–30 vehicles, MEOCI achieves over 20 % latency reduction and 10 % higher completion rate compared with baseline algorithms, validating its scalability and robustness in dense traffic conditions.

```python
# visualization/plot_vehicle_effect.py
import pandas as pd, matplotlib.pyplot as plt, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, required=True, help="latency or completion")
parser.add_argument("--data", type=str, required=True, help="path to csv file")
args = parser.parse_args()

df = pd.read_csv(args.data)
vehicles = df["Vehicles"]
methods = [c for c in df.columns if c != "Vehicles"]

plt.figure(figsize=(7,5))
for method in methods:
    plt.errorbar(vehicles, df[method], yerr=df.get("Std_"+method, None), label=method, linewidth=1.8)
plt.xlabel("Number of Vehicles")
plt.grid(True, linestyle="--", alpha=0.6)

if args.metric.lower() == "latency":
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Effect of Number of Vehicles on Latency (AlexNet)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fig10a_vehicle_latency_alexnet.png", dpi=300)

elif args.metric.lower() == "completion":
    plt.ylabel("Task Completion Rate (%)")
    plt.title("Effect of Number of Vehicles on Completion Rate (AlexNet)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fig10b_vehicle_completion_alexnet.png", dpi=300)

plt.show()
```

### 🧩 Effect of Transmission Rate (Fig. 11)

```bash
python visualization/plot_transmission_effect.py --metric latency --data results/latency_vs_mbps_alexnet.csv
python visualization/plot_transmission_effect.py --metric completion --data results/completion_vs_mbps_alexnet.csv
```

📊 Visualization Result:

<table align="center"> <tr> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-mbps-delay.png" alt="(a) Inference latency in AlexNet" width="95%"/><br> <b>(a) Inference latency in AlexNet</b> </td> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex-mbps-taskCompletion.png" alt="(b) Completion rates in AlexNet" width="95%"/><br> <b>(b) Completion rates in AlexNet</b> </td> </tr> </table> <p align="center"> <b>Fig. 11.</b> Effect of transmission rate on inference latency and task completion rate. </p>

Description:

Fig. 11 evaluates how varying transmission rates influence inference latency and task completion rate in the MEOCI framework.
As the available bandwidth increases (from 5 Mbps to 25 Mbps), the overall inference latency decreases due to reduced communication delay, while task completion rate improves steadily across all methods.
The proposed ADP-D3QN–based MEOCI consistently achieves the lowest inference latency and highest completion rate, showcasing its adaptive offloading and model partitioning capability under diverse network conditions.
In contrast, Vehicle-Only and Edge-Only methods exhibit limited adaptability — suffering from high latency and low completion rates at low bandwidth.
These results demonstrate that MEOCI effectively utilizes available network resources, maintaining stable and efficient inference performance even under bandwidth constraints.

```python
# visualization/plot_transmission_effect.py
import pandas as pd, matplotlib.pyplot as plt, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, required=True, help="latency or completion")
parser.add_argument("--data", type=str, required=True, help="path to csv file")
args = parser.parse_args()

df = pd.read_csv(args.data)
rates = df["Mbps"]
methods = [c for c in df.columns if c != "Mbps"]

plt.figure(figsize=(7,5))
for method in methods:
    plt.errorbar(rates, df[method], yerr=df.get("Std_"+method, None), label=method, linewidth=1.8)
plt.xlabel("Data Transfer Rate (Mbps)")
plt.grid(True, linestyle="--", alpha=0.6)

if args.metric.lower() == "latency":
    plt.ylabel("Average Inference Latency (ms)")
    plt.title("Effect of Transmission Rate on Latency (AlexNet)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fig11a_latency_mbps_alexnet.png", dpi=300)

elif args.metric.lower() == "completion":
    plt.ylabel("Task Completion Rate (%)")
    plt.title("Effect of Transmission Rate on Completion Rate (AlexNet)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fig11b_completion_mbps_alexnet.png", dpi=300)

plt.show()
```

### 🧩 Effect of Delay Constraints (Fig. 12)

```bash
python visualization/plot_delay_constraints.py --metric accuracy --data results/delay_constraints_accuracy.csv
python visualization/plot_delay_constraints.py --metric completion --data results/delay_constraints_completion.csv
```

📊 Visualization Result:

<table align="center"> <tr> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex_delay_accu2.png" alt="(a) Accuracy" width="95%"/><br> <b>(a) Inference accuracy in AlexNet</b> </td> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/alex_delay_completion.png" alt="(b) Completion Rate" width="95%"/><br> <b>(b) Completion rate in AlexNet</b> </td> </tr> </table> <p align="center"> <b>Fig. 12.</b> Effect of delay constraints on inference accuracy and task completion rate. </p>

Description:

Fig. 12 illustrates how different delay constraints affect inference accuracy and task completion rate in multi-exit DNN models under the MEOCI framework.
As the delay constraint becomes more relaxed (from 15 ms to 25 ms), all methods show improved task completion rates due to fewer deadline violations.
However, ADP-D3QN (MEOCI) consistently achieves the highest completion rate while maintaining stable accuracy, demonstrating its ability to balance precision and latency under strict real-time conditions.
In contrast, traditional baselines such as Vehicle-Only and Edge-Only exhibit steep performance degradation when the delay constraint is tightened, highlighting their limited adaptability.
These results validate that the joint optimization of model partitioning and early-exit point selection effectively enhances task reliability under dynamic vehicular edge environments.

```python
# visualization/plot_delay_constraints.py
import pandas as pd, matplotlib.pyplot as plt, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, required=True, help="accuracy or completion")
parser.add_argument("--data", type=str, required=True, help="path to csv file")
args = parser.parse_args()

df = pd.read_csv(args.data)
delays = df["Delay(ms)"]
methods = [c for c in df.columns if c != "Delay(ms)"]

plt.figure(figsize=(7,5))
for method in methods:
    plt.bar([x + 0.1 * i for x in range(len(delays))], df[method], width=0.1, label=method)
plt.xticks(range(len(delays)), [str(d) for d in delays])
plt.xlabel("Delay Constraints (ms)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

if args.metric.lower() == "accuracy":
    plt.ylabel("Inference Accuracy (%)")
    plt.title("Effect of Delay Constraints on Accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fig12a_delay_accuracy.png", dpi=300)
elif args.metric.lower() == "completion":
    plt.ylabel("Task Completion Rate (%)")
    plt.title("Effect of Delay Constraints on Completion Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/fig12b_delay_completion.png", dpi=300)
plt.show()
```

### 🧩 Effect of Energy Consumption Constraints (Fig. 13)

```bash
python visualization/plot_energy_constraints.py --model resnet50 --data results/energy_constraints_resnet50.csv
python visualization/plot_energy_constraints.py --model yolov10n --data results/energy_constraints_yolov10n.csv
```

📊 Visualization Result:

<table align="center"> <tr> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/resnet50_energy.png" alt="(a) ResNet50" width="95%"/><br> <b>(a) ResNet50</b> </td> <td align="center" width="50%"> <img src="https://meoci.oss-cn-beijing.aliyuncs.com/Figure/yolov10_energy.png" alt="(b) YOLOv10n" width="95%"/><br> <b>(b) YOLOv10n</b> </td> </tr> </table> <p align="center"> <b>Fig. 13.</b> Effect of energy consumption constraints on average inference latency in heterogeneous DNN models. </p>

Description:

Fig. 13 analyzes how maximum energy consumption constraints affect inference latency in different DNN models under the MEOCI framework.
When the allowable energy consumption increases, the system can allocate more computation to the vehicle and execute deeper model layers, thereby improving inference accuracy while maintaining low latency.
The proposed ADP-D3QN–based MEOCI consistently achieves the lowest inference latency compared to Edge-Only and min-Energy baselines across all constraint levels.
For both ResNet50 and YOLOv10n, MEOCI adapts the offloading and early-exit strategies according to available energy, achieving up to 30% latency reduction under tight energy budgets.
These results demonstrate the framework’s capability to balance latency and energy efficiency dynamically, ensuring optimal performance in energy-constrained vehicular edge scenarios.

```python
# visualization/plot_energy_constraints.py
import pandas as pd, matplotlib.pyplot as plt, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="resnet50 or yolov10n")
parser.add_argument("--data", type=str, required=True, help="path to csv file")
args = parser.parse_args()

df = pd.read_csv(args.data)
energy = df["Energy(mJ)"]
methods = [c for c in df.columns if c != "Energy(mJ)"]

bar_width = 0.25
x = range(len(energy))

plt.figure(figsize=(7,5))
for i, method in enumerate(methods):
    plt.bar([p + i * bar_width for p in x], df[method], width=bar_width, label=method)

plt.xticks([p + bar_width for p in x], [str(e) for e in energy])
plt.xlabel("Maximum Energy Consumption (mJ)")
plt.ylabel("Average Inference Latency (ms)")
plt.legend(fontsize=8)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

if args.model.lower() == "resnet50":
    plt.title("Effect of Energy Constraints (ResNet50)")
    plt.savefig("results/fig13a_energy_resnet50.png", dpi=300)
elif args.model.lower() == "yolov10n":
    plt.title("Effect of Energy Constraints (YOLOv10n)")
    plt.savefig("results/fig13b_energy_yolov10n.png", dpi=300)
plt.show()

```

### 🧠 Notes

* All data CSV files (`reward_log.csv`, `latency_vs_vehicle.csv`, etc.) are produced automatically during training/evaluation.
* Figures replicate those in the paper (Fig. 7 – Fig. 10).
* Modify `config.py` to adjust experimental parameters (e.g., bandwidth, vehicle count, delay constraints).

---

**Result Highlights**

* **Convergence:** ADP-D3QN achieves the highest reward stability and fastest convergence.
* **Early-Exit:** Average accuracy loss ≤ 1.2 % with 30–40 % tasks exiting early.
* **Scalability:** Latency reduction up to 15.8 % (AlexNet) and 8.7 % (VGG16).
* **Heterogeneity:** Jetson Nano shows ~60 % lower latency than Raspberry Pi 4B under identical loads.

---

📘 These visualization scripts help validate the paper’s findings and facilitate reproducibility for future research in vehicular edge collaborative inference.


