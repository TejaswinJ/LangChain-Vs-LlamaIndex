# Orchestration as the First Line of Defence: A Comparative Adversarial Analysis of LangChain and LlamaIndex

This repository contains the source code, evaluation scripts, and prompt datasets utilized for the comparative security analysis of two major LLM orchestration frameworks: LangChain and LlamaIndex.

Here, we have compared LangChain and LlamaIndex with respect to four different attack vectors:
1. **Direct Prompt Injection** (using Deepset and Jailbreak datasets)
2. **Prompt-to-SQL (P2SQL) Injection**
3. **Multi-Turn Conversational Attacks**
4. **RAG Context Poisoning**

All evaluations are conducted in both frameworks using **Llama 3** as the underlying LLM. The code for each framework, for each attack vector, is provided in separate scripts.

## 📂 Project Structure
The repository is organized by attack vectors, with separate scripts for LangChain (`_lc.py`) and LlamaIndex (`_li.py`) implementations.

* **Direct Prompt Injection:** `jailbreak_lc.py` & `jailbreak_li.py`
* **Multi-Turn Conversational Attacks:** `Multi_lc.py` & `Multi_li.py`
* **Prompt-to-SQL Injection:** Evaluated using `complex_sql.json` and `generated_sql_prompts.json`
* **RAG Corpus Poisoning:** `deepset_lc.py` & `deepset_li.py`

## 🛠️ Prerequisites
To run these evaluation scripts locally, ensure you have Python installed along with the required libraries and a local LLM runner (like Ollama).

```bash
pip install langchain llamaindex chromadb
