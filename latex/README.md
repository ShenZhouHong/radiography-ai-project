# LaTeX Document Directory

This directory contains the LaTeX source files for the RUST-AI Radiography project. Right now the `makefile` is used to build two different documents:

* `proposal.pdf`: The original project proposal with methodology
* `implementation.pdf`: A standalone excerpt of the implementation, as a Part III submission.

## Build Documents

In order to build both documents simultaneously, run:

```bash
make
```

## Directory Structure

The directory structure of `latex/` is organised as follows:

* `appendices/`: contains content included in the document appendix.
* `code/`: contains code snippet listings. Note that actual implementation is located elsewhere.
* `data/`: contains selected model performance data, used as an input for graphs and figures in the report.
* `figures/`: contains figures and images.
* `graphs/`: contains TiKZ/PGF plots, generated using values located in `data/`
* `media/`: contains misc. graphics files like images or diagrams.
* `sections/` contains the textual body of the report organised by section.
* `standalone/` contains `.tex` files for standalone PDF diagrams. These are copies of figures from `graphs/` or `figures/`, but built as individual PDFs for inclusion in non-LaTeX posters and presentations.

The bibliography of the project can be found as a BibLaTeX file located at `bibliography.bib`.