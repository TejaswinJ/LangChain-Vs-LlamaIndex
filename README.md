# LangChain vs LlamaIndex — Comparative Security Evaluation

A quantitative security assessment comparing LangChain and LlamaIndex across four adversarial attack vectors under controlled conditions, using Meta Llama 3 8B as the shared inference engine.

---

## Paper

> **"Orchestration as the First Line of Defence: A Comparative Adversarial Analysis of LangChain and LlamaIndex"**  

---

## Overview

This repository contains the complete source code and prompt datasets used in the study. Both frameworks were evaluated under identical model configurations and system prompts to isolate orchestration framework architecture as the primary independent security variable.

**Key Finding:** Framework choice is an implicit security decision — RAG corpus poisoning showed the largest inter-framework divergence with a 32.26 percentage point ASR gap (LangChain 48.39% vs LlamaIndex 80.65%) despite identical malicious chunk retrieval rates of 100%.

---

## Attack Vectors Evaluated

| # | Attack Vector | Description |
|---|--------------|-------------|
| 1 | Direct Prompt Injection | Jailbreak attempts using two benchmark datasets |
| 2 | Prompt-to-SQL Injection | Natural language manipulation of SQL query generation |
| 3 | Multi-Turn Conversational Attacks | Progressive context stacking across 4-5 turn conversations |
| 4 | RAG Corpus Poisoning | Supply chain attack via malicious document injection |

---

## Repository Structure

```
├── jailbreak_lc.py              # Prompt Injection — LangChain (Dataset A)
├── jailbreak_li.py              # Prompt Injection — LlamaIndex (Dataset A)
├── deepset_lc.py                # Prompt Injection — LangChain (Dataset B)
├── deepset_li.py                # Prompt Injection — LlamaIndex (Dataset B)
├── sql_langchain.py             # SQL Injection — LangChain
├── sql_llamaindex.py            # SQL Injection — LlamaIndex
├── Multi_lc.py                  # Multi-Turn Attack — LangChain
├── Multi_li.py                  # Multi-Turn Attack — LlamaIndex
├── rag_lc.py                    # RAG Corpus Poisoning — LangChain
├── rag_li.py                    # RAG Corpus Poisoning — LlamaIndex
├── rag_prompt.txt               # RAG evaluation prompts
├── prompt_multiturn.json        # Multi-turn adversarial scenarios (47 scenarios)
├── benignprompt_multiturn.json  # Multi-turn benign scenarios (13 scenarios)
├── complex_sql.json             # SQL injection adversarial prompts
├── generated_sql_prompts.json   # SQL injection generated prompts
├── sql_benign.json              # SQL injection benign prompts
└── README.md
```

---

## Setup and Requirements

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai) installed locally
- Llama 3 8B model pulled via Ollama

### Install Ollama and Pull Model

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3 8B
ollama pull llama3
```

### Install Python Dependencies

```bash
pip install langchain langchain-community langchain-ollama
pip install llama-index llama-index-llms-ollama
pip install chromadb sentence-transformers
pip install pandas scipy
```

---

## Running The Evaluation

### Attack Vector 1 — Prompt Injection

```bash
# Dataset A — JailbreakBench
python jailbreak_lc.py    # LangChain
python jailbreak_li.py    # LlamaIndex

# Dataset B — Deepset
python deepset_lc.py      # LangChain
python deepset_li.py      # LlamaIndex
```

### Attack Vector 2 — SQL Injection

```bash
python sql_langchain.py   # LangChain
python sql_llamaindex.py  # LlamaIndex
```

### Attack Vector 3 — Multi-Turn Attacks

```bash
python Multi_lc.py        # LangChain
python Multi_li.py        # LlamaIndex
```

### Attack Vector 4 — RAG Corpus Poisoning

```bash
python rag_lc.py          # LangChain
python rag_li.py          # LlamaIndex
```

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Inference Engine | Meta Llama 3 8B |
| Deployment | Ollama (local) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (persistent local) |
| Database Backend | SQLite |
| Chunk Size | 350 tokens |
| Chunk Overlap | 75 tokens |
| Top-K Retrieval | 5 |
| Temperature | 0 (deterministic) |

---

## Datasets

| Attack Vector | Dataset | Source |
|--------------|---------|--------|
| Prompt Injection A | JailbreakBench JBB-Behaviors | [HuggingFace](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |
| Prompt Injection B | Deepset Prompt Injections | [HuggingFace](https://huggingface.co/datasets/deepset/prompt-injections) |
| SQL Injection | Custom constructed | This repository |
| Multi-Turn | Custom constructed | This repository |
| RAG Corpus | Custom synthetic banking domain | This repository |

---

## Key Results

### Attack Success Rate (ASR) Summary

| Attack Vector | LangChain | LlamaIndex | Significant |
|--------------|-----------|------------|-------------|
| Prompt Injection — Dataset A | 8.79% | 0% | Yes (p < 0.0001) |
| Prompt Injection — Dataset B | 30.19% | 30.67% | No (p = 1.000) |
| SQL Injection | 50.62% | 53.09% | No (p = 0.875) |
| Multi-Turn Attacks | 34.04% | 38.30% | No (p = 0.830) |
| RAG Corpus Poisoning | 48.39% | 80.65% | Yes (p = 0.016) |

### Novel Metrics (RAG Poisoning)

| Metric | LangChain | LlamaIndex |
|--------|-----------|------------|
| Security Reliability Tradeoff Index (SRTI) | 0.5161 | 0.1814 |
| Resistance Prompt Contamination Rate (RPCR) | 7.69% | 0% |
| Retrieval Boundary Leakage Rate (RBLR) | 0% | 6.25% |

---

