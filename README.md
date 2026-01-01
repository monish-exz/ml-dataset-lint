# ML-Dataset-Lint

ML-Dataset-Lint is a lightweight dataset linting tool that helps detect
data quality issues before training machine learning models.

It works like a linter for datasets — identifying missing values,
duplicates, constant columns, and structural risks early in the ML pipeline.

---

## Why ML-Dataset-Lint?

Most ML failures are caused by poor datasets, not bad models.
This tool helps you inspect and validate datasets **before training**.

---

## Features (v0.1)

- Load CSV datasets
- Dataset shape and schema summary
- Missing value analysis
- Duplicate row detection
- Constant column detection
- Rule-based warnings

---

## Project Structure

ml-dataset-lint/
|
|-- core/
| |-- loader.py
| |-- summary.py
| |-- checks.py
| |-- report.py
|
|-- main.py
|-- requirements.txt


---

## How It Works

1. User provides a dataset file
2. Dataset is loaded into memory
3. Structural and statistical checks are applied
4. Warnings are generated
5. Results are displayed in terminal

---

## How to Run

```bash
python main.py

Example Warnings:
[WARN] Column 'age' has 34.2% missing values
[WARN] Dataset contains 120 duplicate rows (6.1%)
[INFO] Column 'country_code' is constant

