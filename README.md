# Assignment 3: Reinforcement Learning from Human Feedback (RLHF)

This repository contains the full implementation and evaluation for **Assignment 3: Reinforcement Learning from Human Feedback (RLHF)**.  
We implement and compare **Reward Modeling**, **PPO-based RLHF**, **GRPO**, and **Direct Preference Optimization (DPO)** using a small pretrained language model.

All experiments are designed to be **reproducible**, **computationally efficient**, and compliant with the assignment requirement to use **low-cost models**.

---

## Overview

Implemented components include:

- Preference data preprocessing using the Anthropic HH-RLHF format
- Pairwise reward model training
- Policy optimization via:
  - PPO (Proximal Policy Optimization)
  - GRPO (Group Relative Policy Optimization)
  - DPO (Direct Preference Optimization)
- Quantitative and qualitative evaluation:
  - Reward scores
  - KL divergence
  - Win-rate comparisons
  - Failure mode analysis

---

## Repository Structure

```
LLM-Assignment-3/
├── data/
│ └── eval_prompts.json # Prompts used for evaluation & win-rate testing
│
├── src/
│ ├── dataset.py # Dataset loading & preprocessing
│ ├── reward_model.py # Reward model definition
│ ├── ppo.py # PPO loss and update logic
│ ├── grpo.py # GRPO implementation
│ ├── dpo.py # DPO objective
│ ├── rl_utils.py # Shared RL utilities
│ └── eval.py # (reserved for extended evaluation)
│
├── train_reward.py # Reward model training
├── train_ppo.py # PPO training script
├── train_grpo.py # GRPO training script
├── train_dpo.py # DPO training script
│
├── eval_models.py # Automatic evaluation (reward, KL)
├── gen_winrate_samples.py # Win-rate sample generation
│
├── ANALYSIS.md # Part 4: Analysis & Evaluation
├── training_curves.png # Training curves visualization
│
├── eval_results.json # Automatic evaluation results
├── winrate_cases.json # Win-rate comparison data
├── winrate_viewer.html # Interactive win-rate viewer
│
├── policy_ppo.pt # Trained PPO policy
├── policy_grpo.pt # Trained GRPO policy
├── policy_dpo.pt # Trained DPO policy
├── reward_model.pt # Trained reward model
│
├── Dockerfile # Containerized environment
└── README.md
```

---

## Environment Setup

### Option 1: Docker (Recommended)

The project is fully containerized for reproducibility.

```bash
docker build -t rlhf-assignment .
docker run --gpus all -it rlhf-assignment
```
Option 2: Local Setup
```bash
pip install torch transformers datasets accelerate tqdm numpy
```
Tested with:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- HuggingFace Transformers

---

## Running the Code
### 1. Train Reward Model
```bash
python train_reward.py
```

Output:
- reward_model.pt
- Training accuracy & loss logs

### 2. Train Policy Models
**PPO**
```bash
python train_ppo.py
```
**GRPO**
```bash
python train_grpo.py
```
**DPO**
```bash
python train_dpo.py
```
Each script initializes from the same supervised base model for fair comparison.

### 3. Evaluation
**Automatic Evaluation (Reward + KL)**
```bash
python eval_models.py
```
Results saved to:
- `eval_results.json`

**Win-rate Evaluation**
```bash
python gen_winrate_samples.py
```
Outputs:
- `winrate_cases.json`
- `winrate_viewer.html` (interactive inspection)

## Results & Analysis

All quantitative and qualitative analyses required in Part 4 are documented in `ANALYSIS.md`.


This includes:
- Reward & KL statistics
- Pareto frontier comparison
- Win-rate results
- Failure mode analysis
- Alignment quality comparison between PPO, GRPO, and DPO

Training dynamics are visualized in: `training_curves.png`