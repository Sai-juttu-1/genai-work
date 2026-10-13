# Generative AI & Agentic Systems Lab

[![CI](https://github.com/Sai-juttu-1/genai-work/actions/workflows/main.yml/badge.svg)](https://github.com/Sai-juttu-1/genai-work/actions/workflows/main.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20management-DE5FE9?style=flat)
![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)

> A practical engineering repository for building and testing Generative AI and Agentic AI systems.

This project focuses on understanding and implementing the core technologies used in modern AI applications, from LLM fundamentals and embeddings to multimodal AI, structured outputs, model providers, and reliable AI workflows.

**Tokens → Embeddings → Models → Providers → Structured Outputs → AI Systems → Agents**

---

## Engineering Focus

- **LLM Foundations** — tokenization, BPE, context windows, token usage, and embeddings.
- **Embeddings** — vector representations, cosine similarity, PCA, and SVD.
- **Model Selection** — capability, latency, throughput, context, and cost.
- **Provider Abstraction** — unified interfaces for Gemini, Groq, OpenAI, and Ollama.
- **Multimodal AI** — image and document processing.
- **Structured Outputs** — Pydantic models and schema-based validation.
- **AI Reliability** — validation, testing, error handling, and reproducible workflows.
- **Engineering Automation** — Python, uv, pytest, Git, and GitHub Actions.

---

## Architecture

```text
                         AI APPLICATION
                              │
                              ▼
                    ┌───────────────────┐
                    │  Application Layer │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         LLM / RAG        Tools & APIs    Multimodal AI
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    AI SYSTEM COMPONENTS
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
          Cloud Models                 Local Models
                │                           │
        Gemini / Groq / OpenAI            Ollama
```

---

## Core Implementations

| Area | Implementation | Tests |
|---|---|---|
| **Tokens & Embeddings** | `s09_tokens.ipynb` · `tokens_utils.py` | `tests/test_s09.py` |
| **Model Landscape** | `s10_model_matrix.ipynb` · `landscape_utils.py` | `tests/test_s10.py` |
| **Provider Routing** | `s11_providers.ipynb` · `providers_utils.py` | `tests/test_s11.py` |
| **Local Models** | `s12_local_models.ipynb` · `local_models_utils.py` | `tests/test_s12.py` |
| **Vision & Structured Output** | `s13_vision_structured.ipynb` · `vision_utils.py` | `tests/test_s13.py` |

---

## Key Engineering Concepts

### Embeddings

Text can be converted into numerical vector representations that allow applications to measure semantic relationships.

```text
Text
  ↓
Embedding Model
  ↓
Vector Representation
  ↓
Similarity / Analysis
```

The project covers cosine similarity, vector spaces, PCA, and SVD.

### Structured Outputs

LLMs produce probabilistic responses, while applications need predictable data.

```text
LLM Response
     ↓
Schema Validation
     ↓
Valid → Application
Invalid → Error / Recovery
```

Pydantic and structured schemas are used to create a reliable validation boundary.

### Cloud and Local Models

The project compares cloud and local inference approaches.

```text
             AI Application
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Cloud              Local
          │                 │
 Gemini / Groq / OpenAI   Ollama
```

---

## Testing & CI

The project uses automated testing with `pytest` and GitHub Actions.

Tests cover:

- Core functionality
- Data validation
- Provider logic
- Structured outputs
- AI-related utilities

API-dependent tests are skipped when credentials are unavailable in CI.

```text
Code Change
    ↓
Git Commit
    ↓
GitHub
    ↓
GitHub Actions
    ↓
Install Dependencies
    ↓
Run pytest
    ↓
Pass / Fail
```

---

## Repository Structure

```text
genai-work/
│
├── .github/
│   └── workflows/
│       └── main.yml
│
├── module01/
│   ├── s09_tokens.ipynb
│   ├── tokens_utils.py
│   │
│   ├── s10_model_matrix.ipynb
│   ├── landscape_utils.py
│   │
│   ├── s11_providers.ipynb
│   ├── providers_utils.py
│   │
│   ├── s12_local_models.ipynb
│   ├── local_models_utils.py
│   │
│   ├── s13_vision_structured.ipynb
│   └── vision_utils.py
│
├── tests/
│   ├── test_s09.py
│   ├── test_s10.py
│   ├── test_s11.py
│   ├── test_s12.py
│   └── test_s13.py
│
├── showcase/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- Git
- uv

### Clone

```bash
git clone https://github.com/Sai-juttu-1/genai-work.git
cd genai-work
```

### Install Dependencies

```bash
uv sync
```

### Run Tests

```bash
uv run pytest
```

---

## Technical Coverage

```text
Generative AI
├── LLM Foundations
├── Tokens & Embeddings
├── Model Selection
├── Provider Abstraction
├── Local Models
├── Multimodal AI
├── Structured Outputs
└── AI Testing

Engineering
├── Python
├── pytest
├── uv
├── Git
└── GitHub Actions

Agentic AI
├── RAG
├── Vector Databases
├── Tool Calling
├── Agent Workflows
├── Agent Evaluation
├── Guardrails
├── MCP
└── AI Observability
```

---

## Future Direction

The project is being extended toward larger Generative AI and Agentic AI systems:

- RAG pipelines
- Vector databases
- Tool calling
- Agent workflows
- Agent evaluation
- Guardrails
- MCP integrations
- API orchestration
- AI observability
- Production-oriented AI services

The long-term goal is to build **modular, testable, and reliable AI systems**.

---

## About

This project represents hands-on work in **AI Engineering, Generative AI, and Agentic AI system development**.

> **Understand → Build → Measure → Validate → Test → Automate**

