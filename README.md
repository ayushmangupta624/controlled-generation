# Controlled Generation of Code-Mixed Text

Code for the paper:

> **Multilingual Controlled Generation And Gold-Standard-Agnostic Evaluation of Code-Mixed Sentences**
> Ayushman Gupta\*, Akhil Bhogal\*, Kripabandhu Ghosh
> IISER Kolkata
> [[arXiv]](https://arxiv.org/abs/2410.10580)

---

## Overview

Code-mixing is the practice of alternating between two or more languages in an utterance, common in multilingual communities. This repository implements **Controlled Generation (CG)**, which:

- Generates semantically equivalent code-mixed sentences from a given English sentence.
- Parameterizes the **Code-Mixing Degree (CMD ∈ [0, 1])** to control how much of the matrix language is replaced with English.
- Emulates real-world code-mixing using word frequencies from social media datasets.

Supported language pairs: **English-Hindi**, **English-Bengali**, **English-Spanish**, **English-French**.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys as environment variables

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

### 3. Download datasets

Place datasets in the following paths (update paths in the script if needed):

| Language Pair    | Path | Source |
|-----------------|------|--------|
| English-Hindi   | `Data/L3Cube_hing_twitter.txt` | [L3Cube-HingCorpus](https://github.com/l3cube-pune/code-mixed-nlp) |
| English-Bengali | `BN_Eng_data/FB_BN_EN_FN.txt`, `TWT_BN_EN_FN.txt`, `WA_BN_EN_FN.txt` | [Patra et al. 2018](https://arxiv.org/abs/1803.06745) |
| English-Spanish | `Spa_Eng_data/mt_spanglisheng/spanglish.txt` | [LinCE Benchmark](https://ritual.uh.edu/lince/) |

---

## Usage

Run sections sequentially as needed:

1. **Dictionary Creation** — builds word frequency dicts from real-world code-mixed data.
2. **Base Creation (Prompt A or B)** — calls an LLM to translate and identify switch points.
3. **Scoring** — assigns each word a replacement-priority score.
4. **Generation** — produces code-mixed sentences at a chosen CMD value.

To change the input sentence or CMD value, update the relevant variables at the top of each section (e.g. `Eng_sentt`, `CMD`).

**Prompt A** requires GPT-4. **Prompt B** works with GPT-3.5-Turbo, GPT-4, and Gemini Pro.

---

## Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{gupta2024multilingual,
  title={Multilingual Controlled Generation And Gold-Standard-Agnostic Evaluation of Code-Mixed Sentences},
  author={Gupta, Ayushman and Bhogal, Akhil and Ghosh, Kripabandhu},
  journal={arXiv preprint arXiv:2410.10580},
  year={2024}
}
```

---
