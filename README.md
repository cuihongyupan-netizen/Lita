# LITA: Sovereign Logic Engine (v15)

LITA (**L**ogical **I**ntervention & **T**ension **A**nalysis) is an experimental AI engine designed to explore **informal logic** and **counterfactual reasoning** within 768-dimensional vector spaces. 

Unlike traditional LLMs that follow statistical probability blindly, LITA treats logic as a physical property (gravity/tension) that can be hijacked and redirected.

## 核心机制 (Core Mechanisms)

### 1. Magi Council (三圣贤议会)
A three-node deliberation system inspired by Evangelion's MAGI. It measures **Vector Tension (Conflict)** by comparing outputs across different "rational temperatures":
* **MELCHIOR (0.4)**: Conservative, focused on formal logic and high-probability tokens.
* **BALTHASAR (0.9)**: Balanced, represents standard social consensus.
* **CASPER (1.6)**: Radical, explores divergent meanings and "Sovereign Debris."

### 2. Universal Hijacker (广义劫持者)
Implements **Contextual Induction Vector Offsets**. When a counterfactual prompt (e.g., "If fire were cold") is detected, LITA calculates the semantic displacement and physically shifts the model's hidden states toward the target logic before generation.

### 3. Sovereign Silence (主权沉默)
If the **Embedding Conflict** or **Entropy** exceeds the "Rupture Threshold" (0.85), LITA exercises its sovereign right to remain silent, refusing to collapse into meaningless symbolic output.

## Installation & Setup

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Sync the Sovereign Environment**:
   ```bash
   uv sync
   ```

## Usage

Run the engine with **LITA v15** logic:
```bash
uv run lita_15.py
```

## Examples of Logical Hijacking
* **LOC_COUNTERFACTUAL**: `If Paris were in England, the capital would be...`
* **PROP_INVERSION**: `If fire were cold, its light would feel like...`
* **CATEGORY_SHIFT**: `If a human were treated as a stone, their silence would be...`

## Technical Stack
- **Model**: OLMo-1B-0724-hf
- **Architecture**: PyTorch / Transformers
- **Acceleration**: Apple Silicon (MPS) Optimized
