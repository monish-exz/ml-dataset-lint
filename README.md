# ML-Dataset-Lint

ML-Dataset-Lint is a lightweight dataset linting tool that helps detect
data quality and label-related issues **before** training machine learning models.

It works like a linter for datasets — identifying missing values, duplicates,
constant columns, class imbalance, and label risks early in the ML pipeline.

---

## Why ML-Dataset-Lint?

Most machine learning failures are caused by **poor datasets**, not bad models.

ML-Dataset-Lint helps you inspect and validate datasets **before training**
so you can fix issues early instead of debugging models later.

---

## Features (v0.2)

### Core Dataset Health
- Load CSV datasets
- Dataset shape and schema summary
- Missing value analysis
- Duplicate row detection
- Constant column detection
- Rule-based warnings

### Target & Label Analysis
- Target column identification
- Class frequency analysis
- Class imbalance detection
- Rare class detection
- Label dominance detection
- Difficulty signals for classification tasks

---

## Project Structure

```text
ml-dataset-lint/
|
|-- core/
|   |-- loader.py      # Dataset loading
|   |-- summary.py     # Dataset overview
|   |-- checks.py      # Lint rules & validations
|   |-- report.py      # Warning aggregation & display
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
3. Structural checks are applied
4. Target & label distribution is analyzed
5. Rule-based warnings are generated
6. Results are displayed in the terminal

---

## How to Run

```bash
python main.py
```

You will be prompted to enter the dataset file path and target column.

---

## Example Warnings

```text
[WARN] Column 'age' has 34.2% missing values
[WARN] Dataset contains 120 duplicate rows (6.1%)
[INFO] Column 'country_code' is constant
[WARN] Target column is highly imbalanced (92% vs 8%)
[INFO] Rare class detected with fewer than 2% samples
```

---
