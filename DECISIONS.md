# Engineering Decisions and Assumptions

## 1. Configuration-Driven Processing

- Source-specific column mappings are maintained in configuration files.
- The processing pipeline should not depend on fixed column names or a
  single source-system format.
- New source formats should be handled through configuration wherever
  possible, rather than changes to the core processing logic.

## 2. Financial Data Handling

- Amounts must be interpreted using the configured gross or net basis.
- Tax adjustments should use available transaction-level information
  and configured defaults where applicable.
- Currency conversion requires a valid exchange rate.
- Missing or invalid financial information should be flagged rather
  than silently converted into misleading spend values.

## 3. Dates and Statuses

- Date fields should be interpreted using configured mappings and
  conservative parsing rules.
- Invalid or ambiguous dates should be flagged.
- Transaction statuses should be interpreted through configurable
  status mappings.
- Records that cannot be reliably interpreted should not be silently
  treated as approved transactions.

## 4. Supplier and Item Resolution

- Normalized text is used to improve matching across inconsistent
  supplier and item descriptions.
- Exact matches are preferred where reliable identifiers are available.
- Fuzzy matching should use conservative thresholds to reduce
  incorrect entity merges.
- Matching decisions should remain traceable in the audit output.

## 5. Exclusions and De-duplication

- Exclusion rules are maintained separately from the processing code.
- Duplicate detection should consider relevant transaction attributes
  and source context.
- Cross-source matching should avoid counting the same underlying
  transaction more than once.
- Excluded and duplicate records should retain traceable reasons.

## 6. Classification

- Predictions must follow the configured category hierarchy.
- Available labelled examples can support supervised classification.
- When labels are unavailable, text-based similarity, embeddings,
  propagation and self-training can provide classification signals.
- Optional LLM assistance may be used when configured credentials
  are available.
- Confidence and classification decisions should remain available
  for review.

## 7. Handling an Unseen Company

- A new company's source files and category hierarchy may differ from
  previously processed inputs.
- The configuration should be drafted from the available file
  structure and validated before a full run.
- If labelled examples are unavailable, classification must rely on
  the configured taxonomy and available text-based signals.
- Unknown or ambiguous fields should be reviewed rather than guessed.

## 8. Reproducibility

- Processing should be executable through the command line using
  configuration files and input data supplied at runtime.
- Randomized model components should use fixed seeds where supported.
- Generated spend and audit outputs should be kept separate from
  source code and raw input data.
