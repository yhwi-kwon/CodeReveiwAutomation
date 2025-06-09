# Automating Code Review Process with LLMs: A Unified Framework for Detection, Generation, and Evaluation

## 1. Project Summary

This repository provides a comprehensive pipeline for automating code review tasks using Large Language Models (LLMs). The system is designed to:

- **Detect the necessity of code review** for a given code diff,
- **Classify the category of code changes** (e.g., bug fix, refactoring, documentation, etc.),
- **Generate category-specific review comments** tailored to the detected change type.

These three core features enable scalable, consistent, and context-aware code review support, making this project suitable for research and practical applications in software engineering.

---

## 2. Pipeline Overview

The following diagram illustrates the end-to-end workflow of the system:

<img width="1107" alt="overview" src="https://github.com/user-attachments/assets/c8412311-f7d0-4396-8070-9877ea80ced8" />


- **Input:** Code diff (patch) in JSONL format.
- **Step 1:** Detect if the code change requires review.
- **Step 2:** Classify the type of code change.
- **Step 3:** Generate a review comment tailored to the change category.
- **Output:** Structured results and metrics for evaluation.

---

## 3. Key Features

### 3.1 Review Necessity Detection (`test_diff_estimation.py`)

- **Purpose:** Automatically determines whether a code change (diff) requires human review using an LLM.
- **How it works:**
  - Reads code diffs from a JSONL file in `data/`.
  - Uses OpenAI API to estimate review necessity and related metrics.
  - Outputs results and evaluation metrics to `output/1.1/`.
- **Example Command:**
  ```bash
  python test_diff_estimation.py
  ```
- **Input Example:** `data/diff_estimation_sampling_100(seed1115).jsonl`
- **Output Example:** `output/1.1/diff_estimation_sampling_100(seed1115).jsonl_gpt-4o-mini_YYYYMMDD_HHMMSS.jsonl`

---

### 3.2 Code Change Category Classification (`test_classification.py`)

- **Purpose:** Classifies the type of code change (e.g., bug fix, refactoring, documentation) using an LLM.
- **How it works:**
  - Reads code diffs from a JSONL file in `data/`.
  - Uses OpenAI API to assign primary, secondary, and tertiary categories and reasons.
  - Outputs results and evaluation metrics to `output/1.2/`.
- **Example Command:**
  ```bash
  python test_classification.py
  ```
- **Input Example:** `data/df_clnl_4.jsonl`
- **Output Example:** `output/1.2/df_clnl_4.jsonl_gpt-4o-mini_YYYYMMDD_HHMMSS.jsonl`

---

### 3.3 Category-specific Comment Generation (`test_codereview.py`)

- **Purpose:** Generates detailed, category-specific code review comments using LLMs, based on the classification results.
- **How it works:**
  - Reads classified code diffs from a JSONL file in `data/`.
  - Uses OpenAI API to generate review comments tailored to the change category.
  - Outputs results to `output/1.3/`.
- **Example Command:**
  ```bash
  python test_codereview.py
  ```
- **Input Example:** `data/df_clnl_4.jsonl_gpt-4o-mini_YYYYMMDD_HHMMSS.jsonl`
- **Output Example:** `output/1.3/df_clnl_4.jsonl_gpt-4o-mini_YYYYMMDD_HHMMSS.jsonl`

---

## 4. Installation & Environment

- **Python version:** 3.8+
- **Install dependencies:**
  ```bash
  pip install -r requirement.txt
  ```
- **OpenAI API Key:**  
  Place your OpenAI API key in a file named `gpt.key` in the project root directory.  
  The first line of this file should contain your API key.

---

## Dataset

- Main data files: `cls-test.jsonl`, `msg-test.jsonl`, `msg-valid.jsonl`
- Source: "Automating Code Review Activities by Large-Scale Pre-Training" (Microsoft, FSE 2022)  
  Zenodo: https://zenodo.org/records/6900648  
  arXiv: https://arxiv.org/abs/2203.09095  
  DOI: 10.5281/zenodo.6900648

- Additional dataset:  
  "Code Review Automation: Strengths and Weaknesses of the State of the Art" (TSE 2024)  
  arXiv: https://arxiv.org/abs/2401.05136  
  Official repository: https://github.com/CodeReviewAutomationSota/code_review_automation_sota

- Other files in the `data/` directory are auxiliary outputs generated for experimentation and evaluation.

For detailed dataset documentation, see [data/README.md](data/README.md).

---

## 5. Experiment Reproducibility

To reproduce the experiments and results:

1. **Prepare Data**

   - Place your input JSONL files in the `data/` directory.
   - Example input files:
     - `diff_estimation_sampling_100(seed1115).jsonl` (for review necessity detection)
     - `df_clnl_4.jsonl` (for classification and comment generation)

2. **Run Each Stage**

   - **Review Necessity Detection:**

     ```bash
     python test_diff_estimation.py
     ```

     - Output: `output/1.1/` directory with result and metrics files.

   - **Code Change Category Classification:**

     ```bash
     python test_classification.py
     ```

     - Output: `output/1.2/` directory with result and metrics files.

   - **Category-specific Comment Generation:**
     ```bash
     python test_codereview.py
     ```
     - Output: `output/1.3/` directory with generated review comments.

3. **Check Results**
   - All outputs are saved in the corresponding `output/` subdirectories.
   - Each stage's output can be used as input for the next stage.

---

For further details, refer to the code and prompt files in the repository.

---

For detailed dataset documentation, see [data/README.md](data/README.md).
