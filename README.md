# EdgeTransAI

**An IoT-Enabled Edge AI Framework for Intent-Aware Multi-Agent Transportation Optimization in Smart Cities**

---
A simulation-based framework for real-time traffic signal control in smart cities, designed for Indian urban traffic conditions (Chennai). Combines edge-deployed Multi-Agent Reinforcement Learning (MARL) with BiLSTM-based vehicle intent prediction and a distributed ADMM optimisation protocol — without depending on cloud infrastructure.
Key features:

🚦 Intent-aware MARL controller using Proximal Policy Optimisation (PPO)
🧠 BiLSTM model predicting vehicle turning intent (straight / left / right / U-turn)
⚡ Edge-local inference at 18.3ms median latency vs 284ms for cloud architectures
🔗 Distributed Optimisation Algorithm (DOA) based on ADMM for multi-node coordination
🛵 Heterogeneous Indian traffic modelling (two-wheelers, auto-rickshaws, buses) using IRC:106-2023 PCU factors
📊 Evaluated on METR-LA, PEMS-BAY, and a Chennai-calibrated SUMO synthetic stream

Results (SUMO simulation):

38% reduction in average intersection delay vs fixed-time control
27.6% improvement in network throughput
Sub-linear latency scaling from 10 to 1,000 edge nodes

Datasets: METR-LA and PEMS-BAY (MIT license, no registration required)
Simulator: SUMO 1.18 (Eclipse Public License 2.0)

This is a simulation study and design blueprint. Physical deployment on Chennai corridors is the planned next step.

## Project Structure

```
EdgeTransAI/
├── train.py                          # PPO training loop
├── evaluate.py                       # Baseline comparison
├── visualize.py                      # Generates all paper figures
├── requirements.txt
├── configs/
│   └── config.yaml                   # All hyperparameters
├── src/
│   ├── agents/
│   │   ├── ppo_agent.py              # PPO actor + spatially-attended critic
│   │   └── doa.py                    # Distributed Optimisation Algorithm (ADMM)
│   ├── models/
│   │   └── intent_model.py           # BiLSTM trajectory encoder, 4-class intent head
│   ├── simulation/
│   │   └── traffic_env.py            # Gymnasium-compatible Dec-POMDP environment
│   └── utils/
│       └── mqtt_bridge.py            # MQTT sensor data pipeline
├── rp/
│   ├── simulation.py                 # Calibrated simulation data for figures
│   └── graphs.py                     # Figure generation (all 7 paper figures)
└── outputs/
    └── checkpoints/
        └── intent_best.pt            # Trained BiLSTM checkpoint
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (quick smoke-test with 10k steps)
python train.py --config configs/config.yaml --steps 10000

# 3. Evaluate
python evaluate.py --config configs/config.yaml

# 4. Generate all 7 paper figures
python visualize.py --all --no-show

# 5. Train intent model independently
python -m src.models.intent_model --train --config configs/config.yaml --epochs 50

# 6. Run synthetic sensor stream
python -m src.utils.mqtt_bridge --synthetic --duration 60
```

---

## VSCode

Open the project folder in VSCode. Five launch configurations are pre-configured in `.vscode/launch.json`:

| Config | Description |
|--------|-------------|
| **Train EdgeTransAI** | Runs `train.py` with 50k steps (quick test) |
| **Generate All Figures** | Produces all 7 paper figures to `outputs/figures/` |
| **Evaluate Agent** | Runs 3 evaluation episodes with a loaded checkpoint |
| **Train Intent Model** | Pre-trains the BiLSTM intent classifier |
| **Synthetic Sensor Stream** | Emits MQTT-style sensor data for 30 seconds |

Press `F5` to launch the selected configuration.

---

## Hardware Deployment Notes

- Edge devices: **NVIDIA Jetson AGX Orin** (32 TOPS) or Jetson Nano (minimal config)
- IoT sensors connect via **MQTT** to the `mqtt_bridge.py` listener
- SUMO is optional — the environment falls back to the numpy stub automatically
- TensorRT INT8 export: add `torch2trt` and call `trt_model = torch2trt(actor, [sample_input])`

---

## Key Results (Simulated)

| Metric | FTC (Baseline) | Cloud DQN | **EdgeTransAI** |
|--------|----------------|-----------|-----------------|
| Avg Delay (s/veh) | 68.4 | 44.7 | **42.1 (−38.4%)** |
| Median Latency (ms) | N/A | 284 | **18.3** |
| Throughput (veh/hr) | 3,240 | 3,970 | **4,134 (+27.6%)** |
| Fuel (L/hr/km) | 0.89 | 0.71 | **0.72 (−19.4%)** |
