# CoWM working paper

Start from [main.tex](main.tex). The English draft follows the action–dynamics coupling discussion in session `01a0ba23-c649-7b90-8acc-96716f894f9f` and the local CoWM notes. Read the [Chinese drafting guide](docs/DRAFT_GUIDE.md) for the argument, evidence boundaries, remaining experiments, and source provenance.

The 2026-10-01 revision fixes six experiment questions and centralizes result slots. Fill [results.tex](results.tex) following the [result filling guide](docs/RESULTS_FILL_GUIDE.md). Conditional rows do not require extending the run queue. CoWM remains a provisional name; online adaptation and C/D/E extensions stay outside the offline core.

## Files

- `sec/0_abstract.tex` through `sec/5_discussion.tex`: main paper.
- `sec/6_appendix.tex`: working supplement, excluded from `main.tex` by default.
- `fig/draft_architecture.tex`: native LaTeX architecture sketch.
- `results.tex`, `tables/final_*.tex`: central result values and six fillable main tables.
- `tables/phase1_7_snapshot.tex`: artifact-derived epoch-10 development checks.
- `sec/7_reference_results.tex`, `tables/phase1_6/`: preserved measured reference results in the supplement.
- `references.bib`: active bibliography.
- `docs/evidence_snapshot.json`: numerical provenance and source SHA256 hashes; internal, not a submission artifact.
- `docs/archive/pre_cowm_20260924.zip`: pre-edit manuscript backup.
- `docs/archive/pre_fillable_20261001.zip`: backup of the 11 files changed in the fillable revision.
- `sec/template_*.tex`, `main.bib`, `fig/teaser.tex`: retained author-kit examples, unused by the manuscript.

## Build

With TeX Live and latexmk installed:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively, upload the project to Overleaf and set `main.tex` as the main document. Use `supplement.tex` to compile the working supplement separately. The `review` mode and anonymous authors are retained. The target year is 2027, but the bundled `cvpr.sty` is from the 2026 author kit and must be checked against the target edition before submission.

The draft contains explicit pending-evidence markers. It is not submission-ready. Local validation checks input files, references, environments, and table provenance; PDF compilation and the eight-page limit are not yet verified because no TeX engine is available in this environment.

## Template provenance

The bundled style originates from the [official CVPR/ICCV/3DV author kit](https://github.com/cvpr-org/author-kit), updated for CVPR 2026. Original examples remain under `sec/template_*.tex`.
