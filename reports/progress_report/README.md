# Progress report (submitted 1 November 2026)

`progress_report.pdf` is the submitted PDF. The rest is its LaTeX source, downloaded from the IE Overleaf
project (BAM template `bam-project.cls`, `[clean,onecolumn]`).

- `main.tex`: metadata, summary, section order. Each section is in `sections/`.
- Figures are in `figures/` (`sophia_roc_pr_curves.png` is in the project but not used).
- Numbers are from before the day t-1 switch (PR #4) and the five-seed CNN (PR #5), so they differ from the
  current `reports/*/README.md`.
- The `texmf/` folder of bundled LaTeX packages (only needed on the IE Overleaf server, which lacks them) is
  not included. On a full TeX Live, `latexmk -pdf main.tex` compiles as is.
