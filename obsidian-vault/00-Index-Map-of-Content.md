---
title: "Map of Content: Modern Computer Chess & Neural Engines"
created: 2026-10-02
tags: [moc, chess, neural-networks, reinforcement-learning, ai]
status: complete
---

# 🧠 Map of Content: The Neural Chess Engine Vault

Welcome to the **Apex Knowledge Vault**, a comprehensive research repository covering the theory, mathematics, architectures, and implementation paradigms of modern computer chess.

```mermaid
flowchart TD
    Index["Map of Content (MOC)"]
    Index --> Hist["01. Evolution & State of the Art"]
    Index --> Data["02. Dataset Engineering & Sources"]
    Index --> NNUE["03. NNUE Architecture Deep Dive"]
    Index --> Trans["04. Transformer & AlphaZero Models"]
    Index --> Search["05. Search Algorithms & Heuristics"]
    Index --> Math["06. Loss Functions & Training Math"]
    Index --> Blue["07. The Cutting-Edge Blueprint"]
    Index --> Bib["08. Landmark Papers Bibliography"]
    Index --> Tax["09. Taxonomy of Top-Level Engines"]
    Index --> FreeData["10. Free Online Datasets Directory"]
    Index --> Weights["11. Pretrained Weights & Papers Archive"]

    Hist --> NNUE
    Hist --> Trans
    Data --> Math
    NNUE --> Blue
    Trans --> Blue
    Search --> Blue
    Blue --> Tax
    Data --> FreeData
    Bib --> Weights
```

---

## 🗺️ Reading Pathways & Structure

### 1. [[01-Evolution-and-SOTA|Evolution and State of the Art]]
The 70-year trajectory of computer chess:
- Claude Shannon's Type-A vs Type-B strategies (1950)
- The Handcrafted Evaluation (HCE) era: Deep Blue, Crafty, Stockfish 1-11
- The Deep Reinforcement Learning rupture: AlphaZero (2017) and Leela Chess Zero (Lc0)
- The NNUE Revolution: Stockfish 12 to 17 (2020–present)
- The Transformer & Searchless era: DeepMind Searchless Chess (2024)

### 2. [[02-Dataset-Engineering|Dataset Engineering & Corpus Acquisition]]
How datasets are mined, filtered, and vectorized:
- Lichess Open Database (billions of rated games and Stockfish evals)
- CCRL / TCEC engine tournament archives
- Stockfish Fishtest self-play dumps (`.bin` and `.binpack` format)
- Syzygy Endgame Tablebases (3-4-5-6-7 man exact WDL and DTZ)
- Feature extraction pipelines: Bitboards, HalfKP, HalfKA_v2, and 8x8x14 spatial tensors.

### 3. [[03-NNUE-Architecture-Deep-Dive|NNUE Architecture Deep Dive]]
The secret behind 100M+ nodes per second:
- Sparse feature representation: HalfKP ($41,024$ inputs) vs HalfKA_v2 ($700,000+$ inputs)
- The Incremental Accumulator: $O(1)$ updates on piece moves
- Activation functions: ClippedReLU and SCReLU (Squared Clipped ReLU)
- Fixed-point SIMD quantization (int8/int16) on AVX2, AVX-512, and ARM NEON
- Dual-network architectures (Stockfish 16/17 Big/Small nets).

### 4. [[04-Transformer-and-AlphaZero-Models|Transformer and AlphaZero Architectures]]
Beyond brute-force search:
- AlphaZero & Lc0 Deep Residual CNNs (20-40 blocks, Squeeze-and-Excitation)
- Multi-Head Self-Attention on 64 chess squares (ViT-style spatial tokens)
- Searchless Chess (Ruoss et al., 2024): 270M parameter decoder-only transformer reaching 2895 Elo
- Policy representations: $1968$ action space vs Move-target coordinates.

### 5. [[05-Search-Algorithms-and-Optimizations|Search Algorithms & Pruning Heuristics]]
Modern tree search techniques:
- Negamax with Alpha-Beta Pruning
- Principal Variation Search (PVS) & Aspiration Windows
- Transposition Tables with Zobrist 64-bit hashing
- Advanced Pruning: Null Move Pruning (NMP), Reverse Futility Pruning (RFP), ProbCut
- Reductions: Late Move Reductions (LMR) based on depth and move count
- Quiescence Search: Tactical stabilization and Static Exchange Evaluation (SEE).

### 6. [[06-Loss-Functions-and-Training-Math|Loss Functions & Training Mathematics]]
The objective functions that train modern engines:
- Win/Draw/Loss (WDL) Cross-Entropy vs Centipawn MSE
- Sigmoid Centipawn mapping: $P(W) = \frac{1}{1 + 10^{-cp / 400}}$
- Policy Cross-Entropy with temperature and visit-count distribution
- Multi-task loss combinations:
  $$\mathcal{L} = \alpha \mathcal{L}_{\text{WDL}} + \beta \mathcal{L}_{\text{Score}} + \gamma \mathcal{L}_{\text{Policy}}$$
- Gradient accumulation, AdamW, and Cosine Annealing schedules.

### 7. [[07-Cutting-Edge-Engine-Blueprint|The Cutting-Edge Engine Blueprint]]
How to build an engine that surpasses Stockfish:
- The Move-Ordering Bottleneck: Replacing heuristic move ordering with a neural policy prior
- Speculative Dual-Engine Architecture: 50M nps NNUE for quiet nodes + Deep Transformer for critical branch points
- Syzygy 7-man knowledge distillation
- End-to-end Python/PyTorch and C++ integration.

### 8. [[08-Landmark-Papers-Bibliography|Landmark Papers & Annotated Bibliography]]
15+ pivotal research papers with links, author summaries, and algorithmic breakthroughs.

---

> [!TIP]
> Use Obsidian's **Graph View** (`Ctrl+G`) to explore connections between mathematical loss functions, feature representations, and search heuristics.
