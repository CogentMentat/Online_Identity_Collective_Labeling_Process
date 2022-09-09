# Online Identity as a Collective Labeling Process: Data and Code
Data and code accompanying article "Online Identity as a Collective Labeling Process" by Alexander T. J. Barron, Marijn ten Thij, and Johan Bollen.

Measurements and figures can be reproduced from the Jupyter notebooks and the data included.  Specific results from our analysis are saved in `.pkl` (python [pickle](https://docs.python.org/3/library/pickle.html)) files, which can be loaded in the notebooks.  Alternatively, slightly different results (from random variation in inference, etc.) can be created by running notebook code and saving new file versions.

Per the Twitter Developer Agreement and Policy, raw tweet and profile data is not allowed to be redistributed, but we do include a file of Twitter User IDs used in our analysis, described below.

Archived at Zenodo: ![DOI](https://zenodo.org/badge/534817418.svg)](https://zenodo.org/badge/latestdoi/534817418)

## Files

All `.bz2` files use level 9 compression.  See notebooks for details.

* `OnlineIdent_CollectiveLabelProc_anchorlabel_profileupdates.json.bz2`: profile updates analyzed in the paper.
* `OnlineIdent_CollectiveLabelProc_TwitterUserIDs.txt.bz2`: The 5,700 Twitter User IDs whose profile updates are analyzed in this work, deemed to be non-bot (see paper).  Current profile data for these users may be harvested through the Twitter API.
* `QuantifyingCollectiveIdentity_nullconspicuousness_precalc.pkl`: Pre-calculated null binned autocorrelation confidence intervals, used as a notebook example.
* `bestparamsim_FRresults.pkl.bz2`: Pre-calculated frequency-rank outputs from the best parameters found in a lattice search (see paper).  The search itself and best parameters can be found by running the notebook.
* `semiotic_tagging_models.py`: Module containing the semiotic tagging models used in the paper.
* `run_semiotic_tag_model.py`: Script for running models, used in conjunction with `semiotic_tagging_models.py`.

## Notebooks

* `Online_Identity_Collective_Labeling_Process_Data_Code.ipynb`: Notebook running simulations and plotting figures from the paper.

## Script requirements (versions used)

* python (3.8.13)
* pandas (1.4.2)
* matplotlib (3.5.1)
* numpy (1.21.5)
* scipy (1.7.3)
