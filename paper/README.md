# CoWM working paper

Start from [main.tex](main.tex). The English draft follows the action–dynamics coupling discussion in session `01a0ba23-c649-7b90-8acc-96716f894f9f` and the local CoWM notes. Read the [Chinese drafting guide](docs/DRAFT_GUIDE.md) for the argument, evidence boundaries, remaining experiments, and source provenance.

The manuscript now focuses on shared A/B learning, three forms of coupling, action selection/refinement, and decision probes. CoWM is a provisional paper name; implementation and checkpoint names remain Fast-LeWAM/R4-AB. Online adaptation and C/D/E extensions are outside the core contribution.

## Files

- `sec/0_abstract.tex` through `sec/5_discussion.tex`: main paper.
- `sec/6_appendix.tex`: working supplement, excluded from `main.tex` by default.
- `fig/draft_architecture.tex`: native LaTeX architecture sketch.
- `tables/`: frozen exploratory table rows extracted from experiment artifacts.
- `references.bib`: active bibliography.
- `docs/evidence_snapshot.json`: numerical provenance and source SHA256 hashes; internal, not a submission artifact.
- `docs/archive/pre_cowm_20260924.zip`: pre-edit manuscript backup.
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
