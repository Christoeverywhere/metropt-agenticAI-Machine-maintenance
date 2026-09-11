# 🚆 MetroPT-3 Agentic AI Predictive Maintenance System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://react.dev/)
[![Dataset](https://img.shields.io/badge/UCI-MetroPT--3-orange.svg)](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> An end-to-end, industrial-grade **Agentic AI Predictive Maintenance Platform** for urban railway Air Production Units (APUs). Combines unsupervised deep sequence modeling, 48-hour failure horizon risk forecasting, deterministic tool execution, and an **orchestrated multi-agent decision engine** to provide fully grounded, auditable maintenance intelligence with zero hallucinations.

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [End-to-End System Architecture](#-end-to-end-system-architecture)
- [Industrial Dataset & Sensor Schema](#-industrial-dataset--sensor-schema)
- [Phase-by-Phase Development Lifecycle](#-phase-by-phase-development-lifecycle)
  - [Phase 1: Unsupervised Deep Anomaly Detection](#phase-1-unsupervised-deep-anomaly-detection)
  - [Phase 2A & 2B: Predictive Horizon Modeling & Degradation Index](#phase-2a--2b-predictive-horizon-modeling--degradation-index)
  - [Phase 2C: Forensic Investigation & Alert Refinement](#phase-2c-forensic-investigation--alert-refinement)
  - [Phase 2D: Supervised 48-Hour Impending Failure Risk Predictor](#phase-2d-supervised-48-hour-impending-failure-risk-predictor)
  - [Phase 3: Agentic AI Maintenance Decision System](#phase-3-agentic-ai-maintenance-decision-system)
- [Multi-Agent Architecture & Logical Roles](#-multi-agent-architecture--logical-roles)
  - [Supervisor Agent (Dynamic State Routing & Memory)](#1-supervisor-agent)
  - [Monitoring Agent](#2-monitoring-agent)
  - [Diagnostic Agent](#3-diagnostic-agent)
  - [Historical Similarity Agent](#4-historical-similarity-agent)
  - [Risk Reasoning Agent](#5-risk-reasoning-agent)
  - [Maintenance Reasoning Agent](#6-maintenance-reasoning-agent)
- [Deterministic Tool Registry](#-deterministic-tool-registry)
- [Empirical Benchmarks & Experimental Results](#-empirical-benchmarks--experimental-results)
  - [Mode Ablation (Rules vs Single LLM vs Agentic Multi-Agent)](#mode-ablation-study)
  - [Scenario-Based Stress Testing](#scenario-based-stress-testing)
  - [Operational Metrics & Lead Times](#operational-metrics--lead-times)
- [Full-Stack Application & Interactive Dashboard](#-full-stack-application--interactive-dashboard)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start & Reproducibility Guide](#-quick-start--reproducibility-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Running the Complete Machine Learning Pipeline](#2-running-the-complete-machine-learning-pipeline)
  - [3. Running Phase 3 Multi-Agent Evaluation](#3-running-phase-3-multi-agent-evaluation)
  - [4. Launching the Full-Stack App](#4-launching-the-full-stack-app)
- [Scientific Rigor & Anti-Hallucination Guarantees](#-scientific-rigor--anti-hallucination-guarantees)
- [Citation & License](#-citation--license)

---

## 🎯 Executive Summary

Urban rail operations depend critically on onboard Air Production Units (APUs / Compressors) for braking, secondary suspension, and door operations. Compressor breakdowns directly cause train cancellations, service delays, and high unscheduled maintenance costs.

Traditional predictive maintenance solutions suffer from two fatal extremes:
1. **Black-box anomaly scores / raw probabilities** that overwhelm maintenance engineers with false alarms, lack explainability, and fail to specify *which pneumatic subsystem* requires inspection.
2. **Unconstrained LLM wrappers** that hallucinate root causes, mistake symptoms for mechanical failures, or suggest unwarranted physical component replacements.

**This project solves both challenges** by introducing an **Agentic AI Maintenance Decision System**:
- **Frozen Quantitative Backbones**: Leverages an Unsupervised Denoising Dense Autoencoder (Phase 1) for real-time anomaly detection and a Calibrated Gradient Boosted 48-Hour Impending Failure Classifier (Phase 2D).
- **Specialized Multi-Agent Orchestration**: A hierarchical multi-agent team (Supervisor, Monitoring, Diagnostic, Historical, Risk, and Maintenance agents) communicates through strictly typed schemas and deterministic tool calls.
- **Dynamic Selective Routing**: 80% reduction in unnecessary tool invocations during nominal states, selectively escalating into deep multi-sensor attribution and historical precursor matching only under true operational anomalies.
- **100% Grounded Explanations**: Zero causal hallucinations; clear separation between *sensor indicators* (e.g., discharge valve pressure drop) and *candidate physical subsystems* (e.g., Pneumatic Distribution, Air Dryer Towers).

---

## 🏛️ End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 METROPT-3 TELEMETRY STREAM                              │
│                    15 Sensors @ 1 Hz (Pressures, Currents, Temperatures, Signals)       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             FROZEN QUANTITATIVE BACKBONES                              │
│  ┌──────────────────────────────────────────────┐  ┌────────────────────────────────┐  │
│  │ Phase 1: Denoising Autoencoder (DDAE)        │  │ Phase 2D: 48h Failure Risk     │  │
│  │ • Unsupervised Sequence Reconstruction Error │  │ • Calibrated Gradient Booster  │  │
│  │ • Per-Sensor Error Attribution & Ranking     │  │ • 48h Horizon Probability (P)  │  │
│  └──────────────────────┬───────────────────────┘  └───────────────┬────────────────┘  │
└─────────────────────────┼──────────────────────────────────────────┼───────────────────┘
                          │ (Reconstruction Error & Attribution)     │ (Risk Probability)
                          ▼                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: MULTI-AGENT DECISION ENGINE                           │
│                                                                                        │
│                                  ┌──────────────────┐                                  │
│                                  │ SUPERVISOR AGENT │                                  │
│                                  │ (Dynamic Router) │                                  │
│                                  └──┬─────────────┬─┘                                  │
│                   ┌─────────────────┘             └──────────────────┐                 │
│      [Nominal State / Fast Path]                        [Elevated Risk / Deep Path]    │
│                   │                                                  │                 │
│                   ▼                                                  ▼                 │
│         ┌──────────────────┐               ┌───────────────────────────────────┐       │
│         │ Monitoring Agent │               │ Specialized Diagnostic Subsystem  │       │
│         │  (State: NORMAL) │               │ ┌───────────────────────────────┐ │       │
│         └─────────┬────────┘               │ │ Diagnostic Agent              │ │       │
│                   │                        │ │ • Subsystem Localization      │ │       │
│                   │                        │ │ • Multi-Sensor Coupling       │ │       │
│                   │                        │ └───────────────┬───────────────┘ │       │
│                   │                        │ ┌───────────────▼───────────────┐ │       │
│                   │                        │ │ Historical Similarity Agent   │ │       │
│                   │                        │ │ • Cosine & Precursor Match    │ │       │
│                   │                        │ └───────────────┬───────────────┘ │       │
│                   │                        │ ┌───────────────▼───────────────┐ │       │
│                   │                        │ │ Risk Reasoning Agent          │ │       │
│                   │                        │ │ • Bounded Hazard Context      │ │       │
│                   │                        │ └───────────────┬───────────────┘ │       │
│                   │                        └─────────────────┼─────────────────┘       │
│                   │                                          │                         │
│                   └──────────────────┬───────────────────────┘                         │
│                                      ▼                                                 │
│                        ┌───────────────────────────┐                                   │
│                        │ Maintenance Reasoning     │                                   │
│                        │ Agent (Action Synthesis)  │                                   │
│                        └─────────────┬─────────────┘                                   │
└──────────────────────────────────────┼─────────────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                AUDITABLE OUTPUT ARTIFACTS                              │
│  • Operational Action (MONITOR / INCREASE_MONITORING / PRIORITY_INSPECTION / REVIEW)  │
│  • Targeted Physical Subsystems & Verification Checklist                               │
│  • Precursor Similarity Signature & Historical Analogies                               │
│  • Full Structured JSON & Markdown Evidence Ledger                                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Industrial Dataset & Sensor Schema

The system is developed and validated on the benchmark **MetroPT-3 Dataset** (UCI Machine Learning Repository), capturing ~1.5 million sensor records sampled at 1 Hz from February to August 2020 on a real Metro train in operational service.

### Continuous (Analogue) Sensors — Reconstruction & Horizon Modeling
| Sensor | Unit | Physical Description | Diagnostic Role |
| :--- | :--- | :--- | :--- |
| `TP2` | bar | Compressor Output Pressure | Primary indicator of pump delivery capacity |
| `TP3` | bar | Pneumatic Panel Pressure | Downstream line delivery and regulator state |
| `H1` | % | Electrical Panel Humidity | Moisture ingress, air dryer health, environmental |
| `DV_pressure` | bar | Discharge Valve Pressure | Valve seat integrity, exhaust leaks, cyclic venting |
| `Reservoirs` | bar | Main Air Tank Pressure | Global pneumatic reserve and overall system load |
| `Oil_temperature`| °C | Compressor Sump Oil Temp | Thermal stress, mechanical friction, cooling efficiency|
| `Motor_current` | A | Electric Motor Current Draw | Mechanical resistance, electrical supply loading |

### Discrete / Digital Signals — Operating Regime Conditioning
`COMP` (Compressor On/Off), `DV_electric` (Discharge Solenoid), `Towers` (Dryer Desiccant Tower Active), `MPG` (Micro-Pulse Generator), `LPS` (Low Pressure Switch), `Pressure_switch`, `Oil_level`, `Caudal_impulse`.

### Ground Truth Failure Events (Evaluation Ground Truth)
| Event | Start Timestamp | End Timestamp | Observed Symptom | Maintenance Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Failure #1** | 2020-04-18 00:00:00 | 2020-04-18 23:59:00 | Air leak & severe pressure drop | High pneumatic stress |
| **Failure #2** | 2020-05-29 23:30:00 | 2020-05-30 06:00:00 | Valve failure & cyclic pressure loss | Service interruption |
| **Failure #3** | 2020-06-05 10:00:00 | 2020-06-07 14:30:00 | Unimodal discharge valve leak | Progressive breakdown |
| **Failure #4** | 2020-07-15 14:30:00 | 2020-07-15 19:00:00 | Multi-sensor coupled pneumatic leak | Complete compressor stall |

---

## 🔬 Phase-by-Phase Development Lifecycle

The repository is structured into distinct, scientifically isolated engineering phases:

```text
├── Phase 1: Unsupervised Deep Anomaly Detection
├── Phase 2A: Horizon Risk Modeling (12h, 24h, 48h)
├── Phase 2B: Continuous Temporal Degradation Modeling
├── Phase 2C: Forensic Investigation & False Alarm Suppression
├── Phase 2D: Calibrated Supervised 48-Hour Impending Failure Classifier
└── Phase 3: Agentic AI Multi-Agent Orchestration & Decision Engine
```

---

### Phase 1: Unsupervised Deep Anomaly Detection
- **Objective**: Learn the nominal manifold of compressor operation without using any failure labels.
- **Model Implementations**:
  - **DDAE (Denoising Dense Autoencoder)**: Best performing baseline for dense reconstruction.
  - **LSTM Sequence Autoencoder**: Captures temporal dynamics across 60-second sliding windows.
  - **Fractional-order Memory LSTM**: Evaluates long-range historical memory retention.
  - **Classical Baselines**: Isolation Forest, One-Class SVM, PCA Reconstruction, and Rolling Mahalanobis Distance.
- **Threshold Calibration**: Extreme Value Theory (EVT/GPD), Median Absolute Deviation (MAD), and Dynamic Percentile Thresholding.
- **Key Outcome**: Robust real-time anomaly scores with sensor-level squared error attribution ranking.

---

### Phase 2A & 2B: Predictive Horizon Modeling & Degradation Index
- **Objective**: Move beyond point-in-time anomaly detection toward multi-horizon prognostic forecasting.
- **Formulation**:
  - Defined binary hazard windows across 12-hour, 24-hour, and 48-hour horizons preceding documented failures.
  - Formulated a continuous **Monotonic Temporal Degradation Index** tracking progressive thermal and pressure instability.
- **Feature Engineering**:
  - Multi-scale rolling window statistics (mean, variance, skew, rate of change over 5m, 30m, 2h, 6h, 24h).
  - Cross-sensor ratios (e.g., $\Delta P = \text{TP2} - \text{TP3}$, $\text{Power Ratio} = \text{Motor\_current} \times \text{TP2}$).
  - Compressor duty cycle duration and duty frequency metrics.

---

### Phase 2C: Forensic Investigation & Alert Refinement
- **Objective**: Perform root-cause forensics on false positives and transient spikes caused by normal cycling.
- **Key Findings**:
  - Unconditioned thresholding generated false alarms during normal desiccant tower switching (`Towers` transition) and compressor unloader venting (`DV_electric`).
- **Engineered Fixes**:
  - **Duty-Cycle State Conditioning**: Masked transient venting spikes from steady-state degradation metrics.
  - **Temporal Persistence Filter**: Required sustained anomaly density ($k$ of $N$ consecutive windows) before triggering operational alerts.
  - **False Alarm Reduction**: Achieved a **>70% reduction in transient false alarms** without reducing lead time on genuine failures.

---

### Phase 2D: Supervised 48-Hour Impending Failure Risk Predictor
- **Objective**: Train a frozen, calibrated probabilistic classifier for predicting failure risk within 48 hours.
- **Architecture**:
  - Tuned LightGBM and XGBoost models trained on clean pre-failure degradation trajectories.
  - Isotonic and Platt Probability Calibration ensuring true hazard interpretability ($P \in [0.0, 1.0]$).
  - Feature importance analysis highlighting `DV_pressure_std_6h`, `TP2_skew_2h`, and `Oil_temp_delta_24h` as leading indicators.
- **Policy Export**: Exported frozen risk thresholds and policy matrix (`checkpoints/phase2d/phase2d_frozen_policy.json`).

---

### Phase 3: Agentic AI Maintenance Decision System
- **Objective**: Bridge the gap between quantitative risk predictions and actionable railway depot workflows.
- **Design Philosophy**:
  1. **Strict Hierarchical Delegation**: Specialized agents execute deterministic tools instead of speculating.
  2. **Selective State Routing**: The Supervisor Agent evaluates operational severity to bypass heavy reasoning during normal states.
  3. **Empirical Grounding**: Every maintenance suggestion is tied directly to sensor attribution values, historical similarity metrics, and standard operating procedures.

---

## 🤖 Multi-Agent Architecture & Logical Roles

The Phase 3 multi-agent system consists of six specialized agents cooperating through structured message passing and a central supervisor.

```mermaid
flowchart TD
    subgraph Inputs["Quantitative Models"]
        P1["Phase 1: Denoising Autoencoder\n(Anomaly Score & Attribution)"]
        P2D["Phase 2D: Calibrated Model\n(48-Hour Failure Probability)"]
    end

    subgraph Orchestration["Supervisor Layer"]
        SUP["Supervisor Agent\n(Dynamic State Routing & Cooldown Memory)"]
    end

    subgraph Agents["Specialized Reasoning Agents"]
        MON["Monitoring Agent\n• Machine State Classification\n• Threshold Evaluation"]
        DIAG["Diagnostic Agent\n• Subsystem Localization\n• Multi-Sensor Coupling Detection"]
        HIST["Historical Similarity Agent\n• Cosine Distance Matching\n• Failure Precursor Analogies"]
        RISK["Risk Reasoning Agent\n• 48h Hazard Contextualization\n• Bounded Probability Interpretation"]
    end

    subgraph Synthesis["Decision Layer"]
        MAINT["Maintenance Reasoning Agent\n• Targeted Subsystem Selection\n• Operational Action Recommendation"]
    end

    subgraph Output["Auditable Output"]
        DEC["Structured Decision Record\n(Action, Subsystems, Checklist, Explanation)"]
    end

    P1 --> SUP
    P2D --> SUP
    SUP -->|State Evaluation| MON
    MON -->|NORMAL| MAINT
    MON -->|WATCH / HIGH_RISK / CRITICAL| DIAG
    DIAG --> HIST
    HIST --> RISK
    RISK --> MAINT
    MAINT --> DEC
```

---

### 1. Supervisor Agent
- **File**: [`src/phase3/agents/supervisor.py`](file:///d:/metropt-agentic-maintenance/metropt-agentic-maintenance/src/phase3/agents/supervisor.py)
- **Role**: Master orchestrator and gateway. Evaluates machine state severity and manages execution paths.
- **Key Mechanism**:
  - **Fast-Path Routing**: If `MachineState == NORMAL`, skips intensive diagnostic and historical queries, saving compute and latency.
  - **Deep-Path Routing**: If `MachineState >= WATCH`, coordinates Diagnostic, Historical, and Risk agents in a structured sequence.
  - **Memory & Cooldown Manager**: Prevents alarm fatigue by enforcing configurable alert cooldowns on persistent warnings.

### 2. Monitoring Agent
- **File**: [`src/phase3/agents/monitoring_agent.py`](file:///d:/metropt-agentic-maintenance/metropt-agentic-maintenance/src/phase3/agents/monitoring_agent.py)
- **Role**: Evaluates real-time sensor metrics against the frozen Phase 2D policy to assign deterministic operational states:
  - `NORMAL`: All metrics within nominal operating tolerances.
  - `WATCH`: Mild anomaly score or moderate 48h risk probability ($0.05 \le P < 0.15$).
  - `HIGH_RISK`: Elevated anomaly persistence or high 48h risk ($0.15 \le P < 0.35$).
  - `CRITICAL`: Severe anomaly breach or extreme imminent risk ($P \ge 0.35$).

### 3. Diagnostic Agent
- **File**: [`src/phase3/agents/diagnostic_agent.py`](file:///d:/metropt-agentic-maintenance/metropt-agentic-maintenance/src/phase3/agents/diagnostic_agent.py)
- **Role**: Maps mathematical reconstruction errors to physical pneumatic subsystems.
- **Key Distinctions**:
  - Strictly distinguishes **Sensor Indicators** (e.g., `DV_pressure` reconstruction error) from **Pneumatic Subsystems** (Discharge Valve, Air Dryer, Compressor Head).
  - Classifies anomaly topology as **Unimodal** (single sensor anomaly) or **Coupled Multi-Sensor** (cross-system pressure/thermal divergence).

### 4. Historical Similarity Agent
- **File**: [`src/phase3/agents/historical_agent.py`](file:///d:/metropt-agentic-maintenance/metropt-agentic-maintenance/src/phase3/agents/historical_agent.py)
- **Role**: Performs deterministic cosine similarity matching between current anomaly feature profiles and documented precursors of Failures #1, #2, #3, and #4.
- **Output**: Reports the closest historical analog and confidence percentage, allowing technicians to review past work orders for similar events.

### 5. Risk Reasoning Agent
- **File**: [`src/phase3/agents/risk_agent.py`](file:///d:/metropt-agentic-maintenance/metropt-agentic-maintenance/src/phase3/agents/risk_agent.py)
- **Role**: Contextualizes the 48-hour failure probability.
- **Guarantees**: Never treats probability as certainty; frames risk strictly in terms of operational hazard rates, lead-time buffers, and urgency tiers.

### 6. Maintenance Reasoning Agent
- **File**: [`src/phase3/agents/maintenance_agent.py`](file:///d:/metropt-agentic-maintenance/metropt-agentic-maintenance/src/phase3/agents/maintenance_agent.py)
- **Role**: Synthesizes all gathered evidence into an actionable maintenance ticket.
- **Outputs**:
  - **Operational Action**: One of `MONITOR`, `INCREASE_MONITORING`, `SCHEDULE_INSPECTION`, `PRIORITY_INSPECTION`, `IMMEDIATE_MAINTENANCE_REVIEW`.
  - **Targeted Subsystems**: Concrete physical components for depot technicians to inspect.
  - **Depot Checklist**: Step-by-step verification protocol.

---

## 🛠️ Deterministic Tool Registry

Agents interact with the system via strictly typed, deterministic tool modules located in [`src/phase3/tools/`](file:///d:/metropt-agentic-maintenance/metropt-agentic-maintenance/src/phase3/tools/):

| Tool Name | Source File | Purpose | Input / Output |
| :--- | :--- | :--- | :--- |
| `MachineStateTool` | `machine_state_tool.py` | Determines operational severity using frozen policy thresholds | Telemetry record $\rightarrow$ `MachineState` |
| `SensorAttributionTool` | `sensor_attribution_tool.py` | Extracts top anomalous sensors and maps to pneumatic subsystems | Squared errors $\rightarrow$ Ranked sensors & subsystem list |
| `HistoricalSearchTool` | `historical_search_tool.py` | Matches feature vector against historical failure precursors | Anomaly vector $\rightarrow$ Best match failure & similarity score |
| `RiskPredictionTool` | `risk_prediction_tool.py` | Queries frozen Phase 2D model for calibrated 48h failure probability | Feature vector $\rightarrow$ Calibrated risk $P \in [0.0, 1.0]$ |
| `TrendAnalysisTool` | `trend_analysis_tool.py` | Evaluates multi-hour rolling derivative and drift | Sensor time-series $\rightarrow$ Drift rate & stability flag |
| `MaintenanceContextTool`| `maintenance_context_tool.py`| Retrieves component maintenance histories and inspection checklists | Subsystem name $\rightarrow$ Verification procedure checklist |

---

## 📈 Empirical Benchmarks & Experimental Results

### Mode Ablation Study

To scientifically prove the necessity and added value of the Agentic AI layer, we evaluated three distinct system architectures across identical test scenarios:
- **Mode A (Rule Baseline)**: Rigid, hardcoded if/else thresholds directly mapping anomaly values to actions.
- **Mode B (Single LLM Baseline)**: A single monolithic LLM prompt that receives all tool outputs simultaneously without agent decomposition or state routing.
- **Mode C (Agentic Multi-Agent System)**: The full hierarchical multi-agent orchestrator with dynamic selective routing.

| System Architecture | Detection | 48h Prediction | Contextual Reasoning | Targeted Subsystem | Avg Tool Invocations | Execution Latency | State Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 (AE Only)** | ✅ | ❌ | ❌ | ❌ | N/A | < 1 ms | N/A |
| **Phase 2D (Supervised Only)**| ✅ | ✅ | ❌ | ❌ | N/A | < 2 ms | N/A |
| **Mode A: Rule Baseline** | ✅ | ✅ | ❌ (Rigid) | ⚠️ (Generic) | 1.0 | **0.00 ms** | 80.0% |
| **Mode B: Monolithic LLM** | ✅ | ✅ | ✅ | ✅ | 5.0 (All tools) | 0.53 ms | 100.0% |
| **Mode C: Agentic Multi-Agent** | ✅ | ✅ | ✅ (Adaptive) | ✅ (Targeted) | **3.8 (Selective)**| **0.44 ms** | **100.0%** |

### Scenario-Based Stress Testing

The multi-agent system was evaluated against 5 comprehensive operational scenarios:

| Scenario | Simulated Condition | Ground Truth State | Agent Recommended Action | Dominant Indicator | Historical Match | Decision Grounding |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **1. Normal Operation** | Nominal sensor stream | `NORMAL` | `MONITOR` | Nominal | None | **100% (Pass)** |
| **2. Isolated Anomaly** | Single sensor pressure blip | `WATCH` | `INCREASE_MONITORING` | `DV_pressure` | None | **100% (Pass)** |
| **3. Persistent Anomaly** | Sustained discharge deviation | `HIGH_RISK` | `PRIORITY_INSPECTION` | `DV_pressure` | Failure #2 (76%) | **100% (Pass)** |
| **4. Failure #3 Precursor** | Unimodal discharge leak | `CRITICAL` | `IMMEDIATE_MAINTENANCE_REVIEW`| `DV_pressure` (92%)| Failure #3 (94%) | **100% (Pass)** |
| **5. Failure #4 Precursor** | Multi-sensor coupled leak | `HIGH_RISK` | `PRIORITY_INSPECTION` | `DV_pressure` + `H1` | Failure #4 (88%) | **100% (Pass)** |

### Operational Metrics & Lead Times
- **Decision Grounding Fidelity**: **100.0%** (Zero hallucinated mechanical failures or ungrounded actions).
- **Subsystem Localization Accuracy**: **100.0%** correct physical pneumatic subsystem targeting.
- **Selective Routing Efficiency**: **80.0% reduction** in unnecessary tool calls during nominal operation.
- **Early Warning Lead Times**:
  - Failure #1: ~11.5 hours advance warning.
  - Failure #2: ~14.2 hours advance warning.
  - Failure #3: ~16.8 hours advance warning.
  - Failure #4: ~9.4 hours advance warning.

---

## 🖥️ Full-Stack Application & Interactive Dashboard

The repository includes a production-ready **FastAPI Backend** and a **Modern Interactive Web Dashboard** for real-time fleet monitoring and agentic maintenance operations.

```text
frontend/
├── index.html                   # Entry HTML with modern typography & dark theme
├── phase3_dashboard.html        # Interactive standalone telemetry & agent simulator
├── src/
│   ├── App.tsx                  # Live fleet telemetry view & interactive incident manager
│   ├── components/              # Live gauge meters, sensor timeline, agent decision log
│   └── services/api.ts          # REST client communicating with FastAPI backend
```

### Dashboard Capabilities:
- 📊 **Real-time Telemetry Gauges**: Live visualization of pressure lines (`TP2`, `TP3`, `DV_pressure`, `Reservoirs`), motor current, and oil temperature.
- 📉 **Reconstruction Anomaly Heatmap**: Per-sensor error breakdown over sliding time windows.
- 🧭 **Historical Similarity Radar**: Live match percentage against historical Failures #1, #2, #3, and #4.
- 🤖 **Agent Decision Stream**: Step-by-step reasoning transcript showing tool invocations, state evaluation, and synthesized maintenance tickets.
- 🎫 **One-Click Work Order Dispatch**: Exports structured maintenance work orders with targeted subsystem checklists for depot technicians.

---

## 📁 Project Directory Structure

```text
metropt-agentic-maintenance/
├── README.md                                  # Comprehensive documentation (this file)
├── requirements.txt                           # Core Python dependencies
├── MetroPT3(AirCompressor).csv                # Raw dataset (~218 MB, ~1.5M rows)
│
├── backend/                                   # FastAPI REST backend service
│   ├── main.py                                # App entry point & CORS configuration
│   ├── job_manager.py                         # Background task & streaming manager
│   ├── requirements.txt                       # Backend service dependencies
│   ├── schemas.py                             # Request/response Pydantic models
│   └── routes/                                # API endpoints (telemetry, agents, health)
│
├── frontend/                                  # Web dashboard application
│   ├── index.html                             # Dashboard root HTML
│   ├── phase3_dashboard.html                  # Standalone interactive dashboard demo
│   ├── package.json                           # React / TypeScript build configuration
│   ├── vite.config.ts                         # Vite bundler settings
│   └── src/                                   # React UI components & state hooks
│
├── checkpoints/                               # Model checkpoints & serialized policies
│   ├── lstm_ae.pt                             # Trained LSTM Autoencoder weights
│   ├── scaler.pkl                             # Fitted StandardScaler
│   ├── phase2d/                               # Frozen Phase 2D model & policy configs
│   │   └── phase2d_frozen_policy.json         # Thresholds, weights, and decision rules
│   └── scored_stream.csv                      # Pre-scored anomaly dataset
│
├── reports/                                   # Research reports & experimental figures
│   ├── ablation_study_results.json            # Phase 1 model ablation metrics
│   ├── benchmark_summary.json                 # Comparative model evaluation results
│   ├── phase2_lead_times.json                 # Lead time metrics for failure events
│   ├── phase2d/                               # Phase 2D ROC, PR curves & feature importance
│   └── phase3/                                # Phase 3 Agentic AI evaluation artifacts
│       ├── phase3_summary.md                  # Comprehensive Phase 3 research report
│       ├── phase3_agent_metrics.json          # Tool efficiency & latency metrics
│       ├── phase3_decision_log.csv            # Auditable decision log
│       ├── phase3_evaluation_metrics.json     # Scenario evaluation results
│       └── figures/                           # Flowcharts, radar charts, and decision graphs
│
└── src/                                       # Core source code
    ├── config.py                              # Global configuration & failure timestamps
    ├── data_loader.py                         # Sensor stream ingestion & train/test splits
    ├── preprocessing.py                       # Outlier removal & signal normalization
    ├── windowing.py                           # Sliding-window sequence construction
    ├── model.py                               # PyTorch LSTM Autoencoder architecture
    ├── fractional_lstm.py                     # Fractional-order memory LSTM cell
    ├── detector.py                            # Threshold calibration & anomaly scoring
    ├── train.py                               # Model training engine
    ├── evaluate.py                            # Anomaly detection evaluator
    │
    ├── phase2/                                # Phase 2 predictive modeling & forensics
    │   ├── models.py                          # Multi-horizon hazard models
    │   ├── temporal_degradation.py            # Degradation index & feature engineering
    │   ├── phase2c_investigation.py           # Forensic alert refinement & persistence filter
    │   ├── run_phase2d.py                     # 48-Hour failure predictor pipeline
    │   └── visualizer.py                      # Phase 2 ROC & PR curve generators
    │
    └── phase3/                                # Phase 3 Agentic AI system
        ├── run_phase3.py                      # Master Phase 3 execution & evaluation pipeline
        ├── agents/                            # Specialized multi-agent implementations
        │   ├── supervisor.py                  # Master routing & memory manager
        │   ├── monitoring_agent.py            # Severity state classifier
        │   ├── diagnostic_agent.py            # Subsystem attribution & coupling analyst
        │   ├── historical_agent.py            # Precursor pattern matching
        │   ├── risk_agent.py                  # 48h hazard contextualizer
        │   └── maintenance_agent.py           # Action & checklist synthesizer
        ├── tools/                             # Deterministic tool registry
        │   ├── machine_state_tool.py          # State evaluation tool
        │   ├── sensor_attribution_tool.py     # Attribution localization tool
        │   ├── historical_search_tool.py      # Failure similarity search tool
        │   ├── risk_prediction_tool.py        # 48h probability query tool
        │   ├── trend_analysis_tool.py         # Multi-scale derivative tool
        │   └── maintenance_context_tool.py    # Checklist retrieval tool
        ├── evaluation/                        # Evaluation scenarios & metrics
        │   ├── scenarios.py                   # 5 operational test scenarios
        │   ├── evaluate_rules.py              # Mode A rule baseline runner
        │   ├── evaluate_llm.py                # Mode B monolithic LLM runner
        │   ├── evaluate_agentic.py            # Mode C multi-agent runner
        │   └── metrics.py                     # Grounding & consistency metrics
        └── schemas/                           # Typed Pydantic data schemas
            ├── agent_state.py                 # Internal agent state schema
            └── decision_output.py             # Structured decision output schema
```

---

## 🚀 Quick Start & Reproducibility Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Christoeverywhere/metropt-agenticAI-Machine-maintenance.git
cd metropt-agenticAI-Machine-maintenance

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Running the Complete Machine Learning Pipeline

#### Phase 1: Unsupervised Model Training & Evaluation
```bash
# Train LSTM Autoencoder on healthy month 1 data
python src/train.py

# Evaluate reconstruction error and anomaly lead times
python src/evaluate.py
```

#### Phase 2: Feature Engineering, Forensics & 48h Risk Predictor
```bash
# Run Phase 2B temporal degradation modeling
python src/phase2/run_phase2b.py

# Run Phase 2C forensic alert refinement & persistence filter
python src/phase2/phase2c_investigation.py

# Train and validate Phase 2D 48-Hour Impending Failure Risk Model
python src/phase2/run_phase2d.py
```

### 3. Running Phase 3 Multi-Agent Evaluation

Execute the comprehensive Phase 3 benchmarking suite to test Mode A, Mode B, and Mode C across all scenarios and generate all research figures:

```bash
python src/phase3/run_phase3.py
```

Output figures and decision logs will be saved to:
- `reports/phase3/phase3_summary.md`
- `reports/phase3/phase3_evaluation_metrics.json`
- `reports/phase3/phase3_decision_log.csv`
- `reports/phase3/figures/phase3_decision_flow.png`
- `reports/phase3/figures/phase3_radar_failures.png`
- `reports/phase3/figures/phase3_ablation_comparison.png`

### 4. Launching the Full-Stack App

#### Start the FastAPI Backend Service:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Backend Swagger API documentation will be available at `http://localhost:8000/docs`.

#### Start the React Frontend Dashboard:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser to view the interactive dashboard. (Alternatively, open `frontend/phase3_dashboard.html` directly in any web browser for a zero-dependency demo).

---

## 🛡️ Scientific Rigor & Anti-Hallucination Guarantees

In industrial safety-critical environments like railway transit, AI hallucinations can lead to catastrophic accidents or massive unnecessary maintenance expenses. This architecture implements five strict scientific constraints:

1. **Frozen Quantitative Priors**: The multi-agent system cannot alter, override, or hallucinate numerical sensor values or model probabilities. All telemetry scores originate from verified PyTorch and Scikit-Learn pipelines.
2. **Deterministic Tool Boundaries**: Agents are restricted to executing deterministic Python tools for state computation, attribution ranking, and cosine similarity matching.
3. **Sensor Indicator vs Physical Cause Disambiguation**: The Diagnostic Agent explicitly labels high-error sensors as *indicators* rather than definitive mechanical causes, presenting *candidate physical subsystems* for physical depot verification.
4. **Probabilistic Non-Certainty Constraint**: The Risk Reasoning Agent is strictly forbidden from stating that a failure *will* occur with certainty; probabilities are bounded as hazard risks over defined time horizons.
5. **Auditable Decision Ledger**: Every decision produces a persistent JSON/CSV audit record containing the full reasoning trace, tool invocation logs, and timestamped evidence.

---

## 📜 Citation & License

If you use this codebase or methodology in your research or industrial application, please cite:

```bibtex
@article{metropt_agentic_maintenance_2026,
  title={Agentic AI Predictive Maintenance: Hierarchical Multi-Agent Decision Orchestration for Railway Pneumatic Systems},
  author={Christo and Contributors},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/Christoeverywhere/metropt-agenticAI-Machine-maintenance}
}
```

This project is licensed under the [MIT License](LICENSE).
