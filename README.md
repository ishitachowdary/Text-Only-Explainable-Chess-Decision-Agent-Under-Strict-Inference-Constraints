# ♟️ Text-Only Explainable Chess Decision Agent Under Strict Inference Constraints

> A compact, CPU-friendly chess decision agent that takes a **text/FEN** board position, proposes a move with a neural model, **enforces legality with an independent rules library**, and returns a **short human-readable rationale** — with all components measured separately.

🌐 **Live frontend demo:** https://cpfe-lilac.vercel.app/

---

## 📑 Table of Contents

1. [Team](#-team)
2. [Project Overview](#-project-overview)
3. [Problem Statement, Aim and Objectives](#-problem-statement-aim-and-objectives)
4. [Theory and Background](#-theory-and-background)
5. [System Architecture](#-system-architecture)
6. [Dataset](#-dataset)
7. [Models](#-models)
8. [Legality-Constrained Move Selection](#-legality-constrained-move-selection)
9. [Explanation Module](#-explanation-module)
10. [Results](#-results)
11. [Verification and Testing](#-verification-and-testing)
12. [Technology Stack](#-technology-stack)
13. [Repository Structure](#-repository-structure-suggested)
14. [Getting Started](#-getting-started)
15. [Limitations and Honest Scope](#-limitations-and-honest-scope)
16. [Future Work](#-future-work)
17. [Responsible AI Disclosure](#-responsible-ai-disclosure)
18. [References](#-references)
19. [Acknowledgements](#-acknowledgements)
20. [License](#-license)

---

## 👥 Team

| Name | University ID | Role / Contribution |
|---|---|---|
| **M. Ishita Sai** | 2300032303 | System design, decision-pipeline coordination and module integration |
| **K. Vijaya Raju** | 2300031302 | Position processing, FEN handling, legal-move generation and chess-rule validation |
| **R. Vaishnavi** | 2300031397 | Move explanation component and explanation-output review |
| **C. Krishna Chaitanya** | 2300031086 | Unit/integration testing, candidate-move validation and issue tracking |

**Faculty Guide:** Dr. Satyabratt Sahoo, Assistant Professor, Department of CSE, KL University

---

## 🔎 Project Overview

A chess decision system must do more than output a plausible move. Its output must be:

1. **Legal** – it must obey the rules of chess.
2. **Bounded** – it must run within small compute/latency budgets.
3. **Understandable** – a human should be able to see *why* the move was chosen.

A neural model that predicts moves without rule checking can return **illegal moves**, and a fluent explanation can be **unfaithful** to the actual decision. This project treats **prediction, legality and explanation as three separate components** that are built and evaluated independently, then combined.

**What the system does**

```
FEN text → parse & validate → encode board → MLP/CNN move scores
        → independent legal-move filter → selected legal move → text rationale
```

**What it is not:** a grandmaster-level engine. It is an experimental research prototype whose value lies in the measurable separation of learned scoring, hard rule enforcement and explanation.

---

## 🎯 Problem Statement, Aim and Objectives

**Problem.** A compact chess decision model may predict an illegal or low-quality move and may give an explanation not grounded in the selected action. Under limited inference resources, an integrated text-oriented pipeline is needed that combines supervised move prediction, independent legality checking, concise explanations and measurable evaluation.

**Aim.** To design, implement and evaluate a resource-conscious chess decision agent prototype that accepts a textual board position, proposes a move, enforces legality independently, and returns a human-readable rationale with transparent performance and quality measurements.

| ID | Objective | Status |
|---|---|---|
| O1 | Prepare a versioned, text-oriented chess dataset from game records | Executed (attach provenance/quality logs) |
| O2 | Build compact supervised move-prediction baselines (MLP, CNN) | Executed, metrics available |
| O3 | Enforce legality and provide grounded textual explanations | Legality diagnostic and explanation prototype reported; explanation evaluation is limited |
| O4 | Evaluate move quality and strict inference constraints | Measured (CPL, engine agreement, latency); full reproducibility logs to be attached |
| O5 | Integrate a reproducible text/API demonstration | Frontend deployed; backend (FastAPI/PostgreSQL) status must be confirmed against the repo |

---

## 📚 Theory and Background

### 1. Chess as an AI testbed
Chess is a deterministic, fully observable, rule-governed game with a very large decision space. A decision agent must (a) interpret the board, (b) know the legal actions, (c) estimate tactical consequences and (d) pick a move that supports a plan. This makes it a clean benchmark for testing whether learned models respect hard constraints.

### 2. Text representations: FEN, PGN, UCI, SAN
- **FEN (Forsyth–Edwards Notation)** – a one-line text encoding of a single position: piece placement, side to move, castling rights, en-passant square, half-move clock and full-move number.
  Example (start position): `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`
- **PGN (Portable Game Notation)** – a text format for whole games (headers + move list). Used here as the training data source.
- **UCI move format** – machine-readable moves such as `e2e4`, `e7e8q` (promotion). Used as the model's label space.
- **SAN** – human-readable moves such as `Nf3`, `O-O`; useful for display.

### 3. Why "text-only"?
Text-only input means the system receives a **structured textual board** (FEN) rather than an image. This removes the need for vision/recognition and keeps the input compact, unambiguous and easy to validate.

### 4. Board-to-tensor encoding
Each position is converted to a binary tensor of shape **12 × 8 × 8**:
- 12 planes = 6 piece types × 2 colours (P, N, B, R, Q, K for White and Black)
- 8 × 8 = board squares
- A square's value is 1 in the plane of the piece standing on it, otherwise 0.

The plane order, square orientation and side-to-move handling must be identical in training and inference.

### 5. Move prediction as classification
Move prediction is framed as **multi-class classification** over a fixed **move vocabulary** (after a mapping correction, **4,208 move IDs**). A model `f` maps the encoded position `x` to a score `s(m | x)` for each move `m`. Training minimises **cross-entropy** between the softmax of these scores and the move actually played in the game (imitation learning).

**Top-k accuracy** = fraction of test positions where the played move is among the model's `k` highest-scoring moves.

> Imitation ≠ optimal play: the label is the move a human chose, not necessarily the best move.

### 6. MLP vs. CNN
| | MLP | CNN |
|---|---|---|
| Input handling | Flattened board vector | Keeps 8×8 spatial layout |
| Strength | Small, simple, fast | Learns local piece patterns and spatial relationships |
| Parameters | 1,344,112 | 2,189,200 |

### 7. Why legality must be enforced outside the model
A softmax over the move vocabulary assigns probability to *every* move ID, including moves impossible in the current position. High score ≠ legal. Therefore the system uses a deterministic rules engine (**python-chess**) as the **authority on legality**:

```
m* = argmax_{m ∈ L(x)}  s(m | x)
```

where `L(x)` is the set of legal moves generated from the current board. If the model's top move is illegal, the highest-scoring **legal** alternative is selected. Legality is only as correct as `L(x)`, so the rules library — not the neural model — generates it.

### 8. Move-quality metrics (engine-relative)
- **Engine agreement** – percentage of positions where the selected move equals the Stockfish reference move.
- **Centipawn loss (CPL)** – evaluation drop (in 1/100 of a pawn) between the engine's best move and the chosen move. Lower is better.
  - *Good move*: CPL ≤ 50
  - *Blunder*: CPL ≥ 300
- Mean CPL is pulled up by rare large errors, so the **median** is reported alongside it.

### 9. Latency metrics
- **Mean latency** – average time per inference call.
- **P95 latency** – the time under which 95% of calls finish; shows tail behaviour.
Raw model inference and the legality-constrained path are reported **separately** because the legality layer adds board processing.

### 10. Explanation and faithfulness
A fluent sentence is not proof of correct reasoning. A good explanation should refer to **observable board features** (centre control, development, king safety, captures, threats, pins) and stay consistent with the selected move. A **perturbation test** changes the input and checks whether the decision changes; this shows *sensitivity*, but **not** semantic correctness — which is why the project flags explanation evaluation as limited.

### 11. Related research
- **ChessLLM / Complete Chess Games Enable LLM Become a Chess Master** – chess as text, supervised fine-tuning for move prediction.
- **Explore the Reasoning Capability of LLMs in the Chess Testbed** – chess reasoning evaluation with strategy/tactic annotations.
- **ChessQA** – evaluation of LLM chess understanding.
- **Grounded Chess Reasoning via Master Distillation** – transferring engine expertise into compact language models.

**Identified gap:** compact move prediction and a dependable text-only decision workflow are rarely combined with an *independent legality constraint* and *separately measured* explanation and inference cost. This project targets that engineering gap — it does **not** claim to beat established engines.

---

## 🏗️ System Architecture

```
 ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
 │ PGN records  │ → │ Parse & extract│ → │ FEN / 12×8×8     │
 │ (training)   │   │ positions      │   │ board encoding   │
 └──────────────┘   └────────────────┘   └────────┬─────────┘
                                                  ↓
 ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐
 │ Text         │ ← │ Selected legal │ ← │ MLP / CNN move   │
 │ rationale    │   │ move (filter)  │   │ scoring          │
 └──────┬───────┘   └────────────────┘   └──────────────────┘
        ↓
 ┌──────────────────────────────────────────────────────────┐
 │ Evaluation harness: accuracy · legality · CPL · latency  │
 │ Stockfish used as an offline reference (not at runtime)  │
 └──────────────────────────────────────────────────────────┘
```

**Design principles**
- Input validation and chess parsing are **deterministic**.
- The neural model supplies **scores, not unconditional actions**.
- The legal-move layer is **authoritative**.
- The explanation module receives the real board and selected move and **must not invent a different move**.
- Stockfish is an **evaluation reference**, not a required runtime dependency.
- Fail closed: malformed FEN is rejected before inference; a missing/incompatible checkpoint is not silently replaced.

| Component | Input | Output | Responsibility |
|---|---|---|---|
| Position parser | FEN/text | Validated board | Syntax and state validation |
| Encoder | Board | 12×8×8 tensor | Consistent model input |
| MLP/CNN predictor | Tensor | Move scores | Learned candidate ranking |
| Legal-move validator | Board + candidates | Legal set | Independent rule enforcement |
| Selector | Legal set + scores | Selected move | Highest-ranked legal move |
| Explanation module | Board + move | Text rationale | Position-linked sentence |
| Evaluation harness | Fixtures, model, engine | Metrics/logs | Accuracy, legality, CPL, latency |
| API / persistence | Request + result | Structured response | Optional service layer |

**Architecture alternatives considered**

| Alternative | Decision |
|---|---|
| Neural model only | Rejected – cannot guarantee legal output |
| Chess engine only | Rejected – misses the compact learned-policy/explainability focus |
| **Hybrid: neural model + legality layer + explanation** | **Selected** |

---

## 🗂️ Dataset

| Item | Value |
|---|---|
| Source | Public PGN game collection (report text cites FICS; Appendix C cites Lichess — **confirm and state one source**) |
| Games | 110,974 standard games (2 untimed-variant games skipped) |
| Extracted positions | 8,255,666 |
| Training split | 6,606,636 positions |
| Validation split | 823,939 positions |
| Test split | 825,091 positions |
| Encoding | `(N, 12, 8, 8)` binary piece planes |
| Label space | 4,208 move IDs |

**Pipeline:** parse PGN game-by-game → filter unsupported variants → for each ply record the position (before the move), side to move and played move → encode board → map move to integer ID → split into train/validation/test.

> ⚠️ Recommended to document: whether the split was at **game level** (to avoid leakage between near-duplicate positions), the random seed, deduplication, licence, and file checksums.

---

## 🧠 Models

| Model | Parameters | Notes |
|---|---|---|
| MLP | 1,344,112 | Dense baseline on the flattened board |
| CNN | 2,189,200 | Spatial filters over the 8×8 board |

- Framework: PyTorch (CPU-only experiments)
- Loss: classification (cross-entropy)
- Both models output scores over the 4,208-move vocabulary.
- **The move-map file must be versioned with the model checkpoint** — prediction indices are only meaningful with the same mapping.

---

## ✅ Legality-Constrained Move Selection

1. Build the board from the FEN with python-chess.
2. Get the model's scores over the move vocabulary.
3. Generate the legal-move set `L(x)` from the rules library.
4. Intersect ranked candidates with `L(x)`; choose the highest-scoring legal move.
5. If no ranked candidate is legal, use a defined safe fallback (score all legal moves, or return a controlled failure). **Never emit an illegal move.**
6. If no legal move exists (checkmate/stalemate), return no move.

---

## 💬 Explanation Module

Produces a concise sentence about observable consequences of the chosen move — e.g. central control, development, king safety, a capture or a direct threat.

> Example: *"The pawn move e2e4 increases control of the center."*

Explanations should be checked against the board before/after the move and, where possible, against engine evaluation or expert review.

---

## 📊 Results

**Environment:** CPU-only, PyTorch (no GPU). Benchmark hardware: Intel Core i5-13500H, 24 GB RAM, Windows 11.

### Model accuracy (held-out test set)

| Model | Train | Validation | Test Top-1 | Test Top-5 |
|---|---|---|---|---|
| MLP | 14.31% | not recorded | 15.83% | – |
| CNN | 17.30% | 19.87% | **20.02%** | **46.34%** |

The CNN beats the MLP by **4.19 percentage points** (top-1).

### Legality diagnostic (CNN)

| Metric | Value |
|---|---|
| Raw top-1 prediction is legal | 58.9% |
| Top-5 contains at least one legal move | 92.19% |
| Legality-constrained accuracy | 28.58% |
| Illegal moves selected | **0%** |

> The 28.58% is **not directly comparable** to the 20.02% unconstrained top-1: the constrained selector changes the candidate set. Always state the protocol and sample count.

### Inference latency

| Path | Mean | P95 |
|---|---|---|
| Raw model | 0.228 ms | 0.276 ms |
| Legality-constrained | 1.334 ms | 3.786 ms |

The legality layer adds board processing and validation, raising latency, but it remains in the low-millisecond range on the tested CPU. Figures are hardware-specific.

### Move quality (10,000 positions, vs. Stockfish)

| Metric | Value |
|---|---|
| Engine agreement | 26.08% |
| Average CPL | 181.87 |
| Median CPL | 43 |
| Good moves (CPL ≤ 50) | 52.68% |
| Blunders (CPL ≥ 300) | 20.09% |

Mean ≫ median CPL indicates a skewed distribution: a subset of large errors inflates the mean.

### Explanation faithfulness diagnostic
A perturbation test over 1,000 positions reported a **99.6% prediction-change rate**. This shows the model is *sensitive* to the perturbation; it does **not** prove the generated sentence is factually correct or causally faithful.

---

## 🧪 Verification and Testing

Recommended and representative test areas:

| Area | Test | Expected |
|---|---|---|
| FEN parser | Valid and malformed FEN | Valid accepted; invalid rejected with clear error |
| Legality | Legal/illegal UCI & SAN | Only legal moves accepted |
| Special rules | Castling, en passant, promotion, check, checkmate, stalemate | Correct legal set |
| Encoder | Known board fixtures | Correct tensor shape and planes |
| Move map | Round-trip move ↔ ID | Consistent; unknown IDs handled safely |
| Model output | Fixed fixture | Correct shape, finite, deterministic scores |
| Explanation | Board + selected move | Refers to the chosen move and a supported feature |
| API (if used) | Missing fields, invalid FEN, no legal moves, checkpoint missing, engine timeout | Structured error responses |

**Acceptance criteria**

| ID | Criterion | Target |
|---|---|---|
| SC-01 | Dataset integrity | Reproducible counts and quality checks |
| SC-02 | Legal move output | Zero illegal outputs on the defined test set |
| SC-03 | Prediction quality | Report achieved top-1/top-5 |
| SC-04 | Explainability | Perturbation + manual review, protocol stated |
| SC-05 | Inference constraints | Hardware, workload, mean and P95 reported |
| SC-06 | API demonstration | Only after reproducible end-to-end evidence |

---

## 🛠️ Technology Stack

| Tool | Purpose |
|---|---|
| Python | Core implementation and data processing |
| PyTorch | MLP/CNN training and inference |
| python-chess | FEN parsing, board state, legal-move generation |
| Stockfish | Reference engine for move-quality evaluation |
| FastAPI | Backend service (development) |
| PostgreSQL + SQLAlchemy | Persistence (development) |
| Vercel | Frontend deployment |
| Git / GitHub | Version control |

**Benchmark environment record:** Windows 11 · Intel Core i5-13500H · 24 GB RAM · Python 3.14.6 · torch 2.13.0 · python-chess 1.11.2 · numpy 2.5.1 · pandas 3.0.5 (Stockfish version to be recorded).

---

## 📁 Repository Structure (suggested)

> Adjust to match your actual repository.

```
.
├── data/
│   ├── raw/                 # PGN files + manifest + checksums
│   └── splits/              # split summary
├── src/
│   ├── preprocess/          # PGN parsing, position extraction
│   ├── encoding/            # FEN → 12×8×8 tensor, move map
│   ├── models/              # MLP, CNN definitions
│   ├── legality/            # legal-move filter & selector
│   ├── explain/             # rationale generation
│   └── api/                 # FastAPI service (if applicable)
├── frontend/                # web UI (deployed on Vercel)
├── artifacts/
│   └── models/              # checkpoints + config + move map
├── eval/                    # accuracy, CPL, latency, faithfulness scripts
├── tests/                   # unit, integration, e2e tests
├── docs/                    # report, figures
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

> Replace the commands below with the project's actual scripts and entry points.

```bash
# 1. Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) Preprocess PGN data
python src/preprocess/build_dataset.py --pgn data/raw/<file>.pgn

# 4. Train
python src/train.py --model cnn

# 5. Run inference on a FEN
python src/predict.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

**Illustrative output** (format from the report's proposed API contract; confirm against the real implementation):

```json
{
  "move": "e2e4",
  "explanation": "The pawn move e2e4 increases control of the center.",
  "is_legal": true,
  "model_version": "<version>",
  "latency_ms": 1.34
}
```

---

## ⚠️ Limitations and Honest Scope

- CNN top-1 accuracy is only **20.02%**; top-5 is **46.34%**.
- Engine agreement is **26.08%**, mean CPL **181.87**, blunder rate **20.09%** — this is **not** a strong chess engine.
- The perturbation score does **not** establish semantic explanation correctness.
- Dataset provenance, split method (game-level?), deduplication and move-label mapping need full reproducibility records.
- Latency figures are specific to the tested CPU/configuration.
- No user study has been done, so **educational benefit is unmeasured**.
- Backend, persistence, security and public deployment of the full stack should be described only as far as evidence supports; the Vercel deployment covers the **frontend only**.
- The associated research paper describes a compact Transformer/LLM fine-tuning plan; the **experiments reported here use MLP and CNN baselines**. Reinforcement learning / preference optimisation are **not** completed and are future work.

---

## 🔭 Future Work

- Game-level splits, deduplication and better label quality.
- Stronger compact architectures and calibrated candidate ranking within fixed resource budgets.
- Better fallback strategies when top-ranked predictions are illegal.
- Board-derived, engine-supported and human-evaluated explanation grounding.
- Complete API integration, DB migrations, end-to-end tests, observability and deployment configuration.
- Controlled user study (SDG 4 – Quality Education linkage is *intended*, not yet demonstrated).
- Compact Transformer/LLM fine-tuning baseline; RL/preference optimisation only with a defined reward and evaluation protocol.

---

## 🤖 Responsible AI Disclosure

ChatGPT was used for drafting, structuring, technical explanation and report preparation. The team verified generated statements, code descriptions, metrics and citations against primary project evidence. No AI-generated citation was retained unless the source was independently located and checked.

---

## 📖 References

1. Y. Zhang et al., "Complete Chess Games Enable LLM Become A Chess Master," NAACL 2025. arXiv:2501.17186
2. S. Wang et al., "Explore the Reasoning Capability of LLMs in the Chess Testbed," NAACL 2025. arXiv:2411.06655
3. Q. Wen et al., "ChessQA: Evaluating Large Language Models for Chess Understanding," arXiv:2510.23948
4. Z. Tang et al., "Grounded Chess Reasoning in Language Models via Master Distillation," arXiv:2603.20510
5. J. Kim et al., "Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation," NAACL 2025
6. E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685
7. python-chess documentation — https://python-chess.readthedocs.io/
8. Stockfish — https://stockfishchess.org/
9. PyTorch documentation — https://pytorch.org/docs/stable/
10. S. J. Edwards, "Portable Game Notation Specification and Implementation Guide."
11. Free Internet Chess Server — https://www.freechess.org/Download/ · Lichess Open Database — https://database.lichess.org/

---

## 🙏 Acknowledgements

Our sincere thanks to our faculty guide **Dr. Satyabratt Sahoo**, and to **Dr. T. K. Rama Krishna Rao** (Principal), **Dr. B. Tirapathi Reddy** (HoD), **Dr. K. Swathi** (Deputy HoD, Student Affairs) and **Dr. R. S. M. Lakshmi Patibandla** (Professor I/C Projects), the Department of CSE and KL University for their guidance and support.

---



<p align="center">Made by Team C2R-18 · KL University · 2026</p>
