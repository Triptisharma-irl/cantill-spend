# SpendCube
### Config-Driven Procurement Spend Analytics & AI Classification

SpendCube is a Python-based data engineering and machine learning project that transforms raw procurement and invoice data into a structured, classified spend cube.

It helps organizations consolidate purchasing records, standardize suppliers and items, handle currencies and taxes, identify duplicate transactions, and classify spend into a company's own procurement category hierarchy.

The pipeline is configuration-driven: company-specific settings are maintained in JSON configuration files rather than hard-coded into the processing logic.

## Key Features

- **Multi-source data ingestion:** Converts different purchasing and ERP exports into a common data structure.
- **Data cleaning:** Handles different number formats, dates, VAT rates, and gross/net amount calculations.
- **Currency conversion:** Uses exchange-rate tables and validates transaction-level FX rates.
- **Supplier and item resolution:** Standardizes inconsistent names and identifiers using normalization, fuzzy matching, and learned aliases.
- **Duplicate detection and matching:** Matches records across purchasing systems to reduce double counting.
- **Procurement classification:** Assigns purchase lines to a custom category hierarchy using machine learning and language-based features.
- **Explainable decisions:** Generates an audit file containing inclusion decisions, exclusion reasons, matching information, and FX details.
- **Config-driven design:** Supports different company data structures through JSON configuration.

## How It Works

```text
Raw Procurement Data
         |
         v
   Data Ingestion
         |
         v
  Cleaning & Parsing
         |
         v
 VAT and FX Processing
         |
         v
 Supplier / Item Resolution
         |
         v
 Duplicate Detection & Matching
         |
         v
 AI-Based Spend Classification
         |
         v
   +-------------------+
   |                   |
   v                   v
Spend Cube CSV     Audit CSV
```

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Fuzzy matching | RapidFuzz |
| Text embeddings | fastembed, ONNX |
| Optional LLM | Anthropic SDK (Claude) |
| Configuration | JSON |
| Version control | Git and GitHub |

## Classification Approach

SpendCube combines multiple classification techniques:

1. **Zero-shot classification signals:** Uses multilingual sentence embeddings and lexical text features to compare purchase descriptions with category labels.
2. **Propagation:** Uses related supplier, item, and account information to improve predictions.
3. **Self-training:** Learns from high-confidence pseudo-labelled examples.
4. **Supervised learning:** Uses available labelled examples to train classification models.
5. **Optional LLM assistance:** Can use Claude for selected high-value, unique purchase descriptions when API credentials are configured.

The final category is selected through a hierarchical, top-down decision process. Each classified record receives a confidence score.

## Project Structure

```text
cantill-spend/
├── configs/              # Company-specific JSON configurations
├── rules/                # Generic processing and exclusion rules
├── scripts/              # Evaluation and matching utilities
├── spendcube/
│   ├── classify/         # Classification models and features
│   ├── ingest.py         # Data ingestion
│   ├── parse.py          # Amount and date parsing
│   ├── clean.py          # Cleaning, VAT and FX processing
│   ├── entities.py       # Supplier and item resolution
│   ├── exclusions.py     # Procurement exclusion rules
│   ├── matching.py       # Cross-system matching
│   └── pipeline.py       # Main processing pipeline
├── README.md
├── APPROACH.md
├── DECISIONS.md
└── requirements.txt
```

## Setup

Requirements: Python 3.10 or newer.

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The embedding model is downloaded on the first run and stored locally. It runs on CPU.

## Running the Pipeline

Provide a JSON configuration and a folder containing the corresponding source files.

```bash
.venv/bin/python -m spendcube \
  --config <path-to-config.json> \
  --data <path-to-company-data> \
  --out outputs/company_A.csv
```

For another company:

```bash
.venv/bin/python -m spendcube \
  --config <path-to-config.json> \
  --data <path-to-company-data> \
  --out outputs/company_B.csv
```

Replace the example paths with the appropriate local dataset locations.

## Output

Each run generates two files:

**1. Spend Cube CSV**

Contains the processed purchase records, including:
- Record identifier
- Inclusion decision and exclusion reason
- Standardized supplier and item keys
- Spend in the reporting currency
- Hierarchical category levels
- Classification confidence

**2. Audit CSV**

Provides additional processing information, including:
- Source and matching details
- Original amount and tax basis
- Currency and exchange-rate information
- Date and amount validation flags
- Exclusion and duplicate decisions

Audit files can contain original source values and should be treated as confidential.

## Useful CLI Options

```bash
--no-labels       Run classification without training labels
--no-embeddings   Skip sentence embeddings
--no-audit        Disable audit output
--train           Provide a training-data directory
--no-labels       Disable label-based classification
```

For all available options:

```bash
.venv/bin/python -m spendcube --help
```

## Optional Claude Integration

For unlabelled company data, the pipeline can optionally use the Anthropic API to classify selected high-spend unique units.

Set `ANTHROPIC_API_KEY` to enable this feature.

To disable LLM calls:

```bash
export SPENDCUBE_NO_LLM=1
```

Without an API key, the pipeline can use its local classification workflow.

## Data Privacy

- Raw procurement datasets are not included in this repository.
- Generated outputs, audit files, caches, and local model artifacts should remain outside version control.
- Audit files may contain sensitive source information.
- Company-specific configurations must be reviewed and sanitized before public distribution.

## Project Goal

The goal of SpendCube is to make procurement spend data more consistent, traceable, and useful for analysis by combining data engineering, entity resolution, machine learning, and optional LLM-based classification.

---

**Built with Python, machine learning, and data engineering.**
