# Technical Approach

## 1. Objective

SpendCube is designed to transform raw procurement records into a
consistent spend dataset classified against a configurable category tree.

The pipeline is designed to work across different source formats and
company-specific configurations without hard-coding individual schemas
into the processing logic.

## 2. Pipeline Architecture

The processing workflow consists of the following stages:

1. **Ingestion:** Read source files using configurable column mappings.
2. **Parsing and cleaning:** Normalize dates, amounts, currencies and text.
3. **Tax and FX processing:** Standardize spend values and flag invalid
   or incomplete financial information.
4. **Entity resolution:** Normalize supplier and item descriptions and
   identify likely matches.
5. **Exclusion and de-duplication:** Apply procurement rules and identify
   duplicate records across sources.
6. **Classification:** Assign eligible purchase records to the configured
   category hierarchy.
7. **Output and audit:** Generate a structured spend output and a separate
   audit trail.

## 3. Classification Strategy

The classification workflow combines multiple signals:

- Text-based features and multilingual sentence embeddings.
- Similarity between purchase descriptions and category descriptions.
- Information propagated from related records and resolved entities.
- High-confidence pseudo-labelled examples through self-training.
- Supervised classification using available labelled examples.
- Optional LLM assistance for selected records when API credentials
  are configured.

Classification follows the configured category hierarchy, and the
pipeline records confidence information alongside predictions.

## 4. Data Quality and Traceability

The pipeline aims to avoid silently converting invalid financial
information into valid-looking spend.

It applies validation and records flags for issues such as:

- Invalid or ambiguous dates.
- Missing or unreliable exchange rates.
- Invalid amounts.
- Excluded or cancelled records.
- Duplicate and cross-source matches.

The audit output provides additional information to understand
processing decisions.

## 5. Configuration-Driven Design

Company-specific source mappings, reporting settings, taxonomy references
and processing options are maintained separately from the Python
processing logic.

This separation makes it possible to adapt the pipeline to a new
company and category hierarchy through configuration rather than
rewriting the core application.

## 6. Tools and Technologies

- Python
- pandas and NumPy
- scikit-learn
- RapidFuzz
- fastembed and ONNX
- JSON configuration
- Optional Anthropic API integration
- Git and GitHub

AI tools and libraries are used for language-based classification
signals and optional LLM assistance. The core processing workflow
also supports local execution.

## 7. Reproducibility

The pipeline is designed to run from command-line arguments and
configuration files, with generated outputs stored separately from
source code and input datasets.

Raw task datasets and generated audit files are not included in
the public source repository.
