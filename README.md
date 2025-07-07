# Ranking phase fields by likelihood with ICSD patterns

13.08.2020 Andrij Vasylenko
8.11.2023 Features Branch
8.06.2025 Refactor

![TOC](https://github.com/lrcfmd/PhaseFieldRanking/blob/master/TOC.png)

## Citing
This submission is a part of the research 

A. Vasylenko et al. "Element selection for crystalline inorganic solid discovery guided by unsupervised machine learning of experimentally explored chemistry", Nat. Commun. 12, 1-12 (2021).

When using this code or any of the numerical results in this repository, please cite the above paper approriately.

## Introduction

In materials science, selection of 118 chemical elements to combine ultimately determines whether a elemental combination (called phase field) can form a stable compound. The combinatorial challenge is overwhelming, e.g., there are 118! / (3! * 115!) = 266916 possible ternary phase fields, with only a fraction of them explored synthetically. While scientists with accumulated knowledge of chemistry of materials can argue, that many combinations can afford stable compounds, depending on the particular stoichiometric ratios, it is still impossible to prioritise the likely attractive combinations.

This model (PhaseRank) affords numerical assesment of the phase fields based on the knowledge exctracted with unsupervised learning of the Inorganic Crystal Structure Database (ICSD) of materials, and ranks the unexplored chemistry by similarity with the materials found in ICSD.

## Requirements

python-3.8

pip (version 19.0 or later)

OS:

Ubuntu (version 18.04 or later)

MacOS (Catalina 10.15.6 or later)


## Dependencies:

TensorFlow (version 2.0 or later)

PyOD (0.7.8 or later)

scikit-learn-0.20.0

Dependencies can be installed automatically during installation.

## Installation
`pip install .`

## Usage
`python -m ranking_phase_fields` <input_file>

E.g., `python -m ranking_phase_fields config.yaml`

## Functionality
Unsupervised pattern recognition methods utilise PyOD library.

Models:

    'AE'             : AutoEncoder
    'VAE'            : Variational AE
    'VAE_encoder'    : Variational AE with extracting latent features

Read more about these methods https://www.pyod.readthedocs.io

[1] Zhao, Y., Nasrullah, Z. and Li, Z.,
PyOD: A Python Toolbox for Scalable Outlier Detection. 
Journal of machine learning research 20(96), pp.1-7 (2019).

Supported elemental features.
When describing element one can select precalculated features
from data/elemental_features/ folder,
including LEAFs - locan environment atomic features
[2] A. Vasylenko et. al. Digital features of chemical elements extracted from local geometries in crystal structures, Digital Discovery 4, 477 (2025)

## Development & Tests

- Example test inputs and example outputs are included in the repository for reference.
- You can run unit tests using `pytest tests/`.

## Parameters of the input file 
(default values are in the config.yaml file)

 parameter | value 
---|--- 
 *icsd_file*    | (default: icsd2017) ICSD excerpt. A text file, a list of ICSD .cif files with specified oxidation states for each element.
*phase_fields*  | (default: quaternary) binary, ternary, quaternary - are supported. Type of phase fields to investigate.
*cations_train* | (default: all) list of elements constituting a phase field in ICSD. Elements in the first positions (cations), e.g. elements for M and M' in MM'AA' phase fields.
*anions_train*  | (default: S,O,Cl,Br,F,N,Te,P,Se,As,I) list of elements constituting a phase field in ICSD. Elements in the last positions (anions) e.g. elements for A and A' in MM'AA' phase fields. 
*nanions_train* | (default: 2) Number of anions (elements with negative oxidation states as specified in icsd_file) in the training set. Supported values: 0, 1, 2. If 0 - oxidation states are not taken into account.
*cation1_test*  | (default: Li) list of elements for the first position (e.g. M in MM'AA' phase fields) in the phase fields to explore (no reported associated compositions in ICSD). 
*cation2_test*  | (default: all) list of elements for the second position (e.g. M' in MM'AA' phase fields) in the phase fields to explore (no reported associated compositions in ICSD). Ignored for binary and ternary (type MAA') phase fields.
*anion1_test*   | (default: S,O,Cl,Br,I,F,N) list of elements for the 3rd position (e.g. A in MM'AA' phase fields) in the phase fields to explore (no reported associated compositions in ICSD). Stands for a 3rd cation in ternary (type MM'M") and quaternary (type MM'M"A) phase fields.
*anion2_test*   | (default: S,O,Cl,Br,I,F,N) list of elements for the 4th position (e.g. A' in MM'AA' phase fields) in the phase fields to explore (no reported associated compositions in ICSD). Ignored for ternary (type MM'A) phase fields.
*method*        | (default: VAE) See all supported models above.
*cross-validate*| (default: Fault) If True sets 5-fold cross-validation of the model.
*average_runs*  | (default: 1) Number of runs to average the scores over. Makes sense for not neural network based (AE, VAE) methods.
*features*      | (default: See config.yaml). See all supported features in data/elemental_features/ folder


## Project structure

- ranking_phase_fields/     ← Main package
- data/                     ← Input data and features (e.g., elemental features, ICSD files)
- tests/                    ← Unit tests
- examples/                 ← Jupyter notebooks and example CSVs

Other files:
- config.yaml               ← Default configuration file with input parameters
- pyproject.toml            ← Build system & dependencies
- README.md                 ← This file
