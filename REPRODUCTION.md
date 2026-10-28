# Reproducing the public release

## 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # add your own API keys
python main.py --check-config
```

The corpus is not redistributed. Place it under `corpus/` before running tools
that depend on it. The expected paths are defined in `src/main.py` and
`src/agent/validation.py`.

## 2. Query sets

- `queries/layer/evaluation_queries_curated.json` — 54-query evaluation set.
- `queries/layer/curated_queries.txt` — the same 54 questions as plain text
  (`query_id<TAB>question`).
- `queries/layer/evaluation_queries_pool195.json` — 195-query candidate pool.
- `queries/layer/pool195_queries.txt` — the same 195 questions as plain text.

The internal regression test suite is not part of this release because it
depends on the non-redistributed corpus.

## 3. Run commands

Single query:

```bash
python main.py --query "What is the crude protein of maize?"
```

Evaluation on the curated set:

```bash
python main.py --evaluate --query-file queries/layer/evaluation_queries_curated.json
```

Multi-model comparison:

```bash
python main.py --compare --query-file queries/layer/evaluation_queries_curated.json
```

Ablation run:

```bash
python main.py --ablation --query-file queries/layer/evaluation_queries_curated.json
```

Outputs are written under `logs/` in the working directory. `logs/` is not
tracked by git.

## 4. What is not reproducible from this release alone

The layer-hen corpus is not redistributed because some source terms do not
clearly permit redistribution. Raw expert workbooks, expert identifiers,
name/label maps, coding keys, full answer keys, per-query expert scores,
internal audit reports, and audit logs are not part of the public release.
De-identified scores and audit records are available to the editor and reviewers
on reasonable request, subject to confidentiality and privacy requirements. See
`DATA_ACCESS.md` for the complete access map.
