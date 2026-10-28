# NutriAgent public release

NutriAgent is a specialist-routed multi-agent system for layer-hen nutrition
question answering. Four domain specialists (Feed, Nutrition, Performance,
Terminology) use isolated tools and a shared retrieval layer; a synthesis stage
checks cross-specialist consistency and integrates their outputs.

This release contains the source code, configuration, prompts embedded in the
source, and the 54-query evaluation set with the 195-query candidate pool. The
query sets are provided as JSON manifests and as plain-text question lists. It
is limited to reusable code and query data.

## Contents

- `src/` — agents, tools, retrieval, evaluation, and runtime code.
- `config/` — model, experiment, and pricing configuration; no API keys.
- `queries/layer/` — the 54-query curated set and the 195-query candidate pool,
  in JSON and plain-text form.
- `DATA_ACCESS.md` — public, controlled-access, and non-redistributed items.
- `REPRODUCTION.md` — install and run commands.

## Not included

This public release does **not** contain:

- the layer-hen corpus, because some source terms do not clearly permit redistribution;
- raw expert workbooks, expert identifiers, name/label maps, or coding keys;
- expert score tables, full answer keys, internal audit reports, audit logs, or generated annotations;
- response/cover letters or manuscript files;
- the internal regression test suite, which depends on the non-redistributed corpus.

The corpus can be placed under `corpus/` using the sources documented in the
manuscript's Table S3. The code that requires corpus files will not run until
the corpus is present.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # add your own API keys
python main.py --check-config
```

See `REPRODUCTION.md` for query and evaluation commands.

## Data and licensing

Code is released under the MIT License (see `LICENSE`). The 54-query and
195-query sets are provided for research use. The layer-hen corpus is derived
from public feed tables and breeder/company documentation; per-breed provenance
and source terms are documented in the manuscript and its Table S3. Third-party
data remain subject to their original terms.

De-identified per-query expert scores and audit materials are not part of this
public release. They are available to the editor and reviewers on reasonable
request, subject to confidentiality and privacy requirements. See
`DATA_ACCESS.md` for the complete access map.

## Citation

If you use this repository, please cite the NutriAgent manuscript (citation
details will be updated after publication).
