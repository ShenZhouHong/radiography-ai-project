# Goldsmiths Final Project
Final Project for Shen Zhou Hong's 2022 BSc in Computer Science at Goldsmiths, University of London.

## Repository Index

### For the report, see `latex/`.

```
latex/
├── appendices
│   ├── appendix.tex
│   └── presentation.tex
├── code
│   ├── cross-validate.tex
│   ├── kfold-dataset.tex
│   ├── model-definition.tex
│   └── regime_I.tex
├── data
│   ├── hypersearch
│   │   ├── regime-1-examples/
│   │   ├── regime-1.csv
│   │   └── regime-2.csv
│   └── initial-evaluations
│       ├── inceptionv3_end2end_train_auc.csv
│       ├── inceptionv3_end2end_valid_auc.csv
│       ├── inceptionv3_imagenet_train_auc.csv
│       ├── inceptionv3_imagenet_valid_auc.csv
│       ├── inceptionv3_radimagenet_train_auc.csv
│       ├── inceptionv3_radimagenet_valid_auc.csv
│       ├── lenet1998_train_auc.csv
│       └── lenet1998_valid_auc.csv
├── figures
│   ├── pdf-spread-kim.tex
│   ├── pdf-spread-lindsey.tex
│   ├── pdf-spread-mura.tex
│   └── protocol-diagram.tex
├── graphs
│   ├── hypersearch-regime-1-examples.tex
│   ├── hypersearch-regime-1.tex
│   ├── hypersearch-regime-2.tex
│   ├── inceptionv3-end2end.tex
│   ├── inceptionv3-imagenet.tex
│   ├── inceptionv3-radimagenet.tex
│   ├── lenet1998.tex
│   └── regime-1-best-candidate.tex
├── media
│   ├── kim-and-mackinnon.pdf
│   ├── lindsey-et-al.pdf
│   ├── metrc-ai-presentation.pdf
│   ├── mura.pdf
│   └── protocol-diagram.pdf
├── sections
│   ├── background.tex
│   ├── bibliography.tex
│   ├── implementation.tex
│   ├── introduction.tex
│   ├── methodology.tex
│   ├── outcomes.aux
│   ├── outcomes.tex
│   └── resources.tex
├── bibliography.bib
├── implementation.pdf
├── implementation.tex
├── proposal.pdf
├── proposal.tex
├── makefile
└── README.md
```

### For the implementation, see `python/`

```
python/
├── analysis
│   ├── regime-1-analysis.ipynb
│   ├── regime-1.csv
│   ├── regime-1-raw-data.pickle
│   ├── regime-2-analysis.ipynb
│   ├── regime-2.csv
│   ├── regime-2-raw-data.pickle
│   └── README.md
├── common
│   ├── __init__.py
│   ├── crossvalidate.py
│   ├── datasetutils.py
│   ├── kfold.py
│   ├── model.py
│   ├── plotting.py
│   ├── utilities.py
│   └── README.md
├── hyperparam-search
│   ├── regime-1.ipynb
│   └── regime-2.ipynb
├── initial-evaluation
│   ├── inceptionv3-end2end.ipynb
│   ├── inceptionv3-imagenet.ipynb
│   ├── inceptionv3-radimgnet.ipynb
│   └── lenet1998.ipynb
└── README.md
```

### For the dataset, see `dataset/`

```
dataset
├── ds_test/
├── ds_train/
├── ds_valid/
├── df_labels-onehot.csv
├── df_labels-onehot.pickle
├── build-tfdataset.ipynb
├── filenames.csv
├── raw_images
└── README.md
```