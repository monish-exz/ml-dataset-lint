# ML-Dataset-Lint

ML-Dataset-Lint is a lightweight, rule-based dataset linting tool that helps detect
data quality, target, and structural risks **before** training machine learning models.

It works like a linter for datasets — identifying missing values, duplicates,
constant columns, class imbalance, leakage signals, bias indicators, and
basic trainability risks early in the ML pipeline.

---

## Why ML-Dataset-Lint?

Most machine learning failures are caused by **poor datasets**, not bad models.

ML-Dataset-Lint helps you inspect and validate datasets **before training**
so you can fix data issues early instead of debugging models later.

This tool is designed to be:
- Simple to run
- Easy to understand
- Useful for students, learners, and early ML projects

---

## Features (v0.6)

### Core Dataset Health
- Load CSV datasets
- Dataset shape and schema summary
- Missing value analysis
- Duplicate row detection
- Constant column detection
- Rule-based warnings

### Target & Label Analysis
- Target column validation
- Class frequency and percentage analysis
- Class imbalance detection
- Rare class detection
- Label dominance signals
- Difficulty indicators for classification tasks

### Correlation & Leakage Signals
- Feature–target correlation checks (numeric)
- High-correlation feature warnings
- ID-like column detection
- Potential data leakage risk signals

### Bias Signals (Heuristic-Based)
- Detection of sensitive categorical features
- Under-represented group warnings
- Representation risk indicators (non-judgmental)

### Dataset Health & Trainability
- Small dataset warnings
- Feature-to-sample ratio checks
- Missing-value readiness checks
- Numeric feature availability checks

---

## Supported Input Format

- **CSV files only** (current version)

Support for additional formats (JSON, TSV, etc.) is planned for future versions.

---

## Project Structure

```text
ml-dataset-lint/
|
|-- core/
|   |-- loader.py      # Dataset loading logic
|   |-- summary.py     # Dataset overview & schema
|   |-- checks.py      # Lint rules and validations
|   |-- report.py      # Warning aggregation & output
|
|-- main.py            # Entry point
|-- requirements.txt
|-- README.md
|-- LICENSE
```

---

## How It Works

1. User provides a dataset file path
2. Dataset is loaded into memory
3. Structural and statistical checks are applied
4. Target and label distributions are analyzed (if provided)
5. Leakage, bias, and trainability signals are evaluated
6. Rule-based warnings are generated
7. Results are displayed in the terminal

---

## How to Run

```bash
python main.py
```

You will be prompted to enter:

Dataset file path

Target column name (optional)
---

## Example Warnings

```text
[WARN] Column 'age' has 34.2% missing values
[WARN] Dataset contains 120 duplicate rows (6.1%)
[INFO] Column 'country_code' is constant
[WARN] Label 'yes' dominates dataset (92.00%)
[WARN] Label 'maybe' has only 12 samples
[WARN] Feature 'user_id' highly correlated with target (possible leakage)
[INFO] Dataset is very small (<100 rows)
[INFO] Missing values must be handled before training
```

---


## Philosophy

ML-Dataset-Lint does not try to make decisions for you.

It provides signals, warnings, and context so you can make better
data decisions before model training.

This tool is intentionally:

Rule-based (not model-dependent)

Explainable

Beginner-friendly

--- 