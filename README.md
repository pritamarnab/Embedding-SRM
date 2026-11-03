# Embedding-SRM

Recent work shows that functional alignment via a shared response model (SRM; Chen et al., 2015) improves encoding and supports cross-subject generalization, but existing SRM methods require participants to experience the same stimulus. Here, we introduce Embedding-SRM (eSRM), which applies SRM to each subject’s encoding weight matrix, eliminating the need for shared stimulus and enabling alignment across disjoint recordings. We also developed a framework using eSRM that allows us to pool neural data from other subjects to effectively enlarge the training set for fitting encoding models in a low-data target participant.

## Installation

Install BrainIAK wheels from PyPI on Linux, macOS, and Windows, e.g. using pip:

`python3 -m pip install brainiak`

If you want to install the environment used for these analyses (on Linux), clone the repo and then use the following command:

`conda env create -f environment.yml`

For Mac users, please see the detailed instructions for installing BrainIAK here:
 https://brainiak.org/docs/installation.html

 ## Usage

 For the Podcast data use [`esrm_podcast.ipynb`](esrm_podcast.ipynb)

 For 24/7 data: follow the instructions in the [makefile](Makefile)


