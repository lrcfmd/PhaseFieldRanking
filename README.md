# Ranking phase fields by likelihood with ICSD patterns

Andrij Vasylenko

## Requirements

python-3.8

pip (version 19.0 or later)

OS:

Ubuntu (version 18.04 or later)

MacOS (Catalina 10.15.6 or later)


## Dependencies:

TensorFlow (version 2.0 or later)

PyOD (0.7.8 or later)

Dependencies can be installed automatically during installation.

## Installation
`pip install .`

## Usage
`python ranking_phase_fields.py` <input_file>

If no input file provided the run is equivalent to
`python ranking_phase_fields.py rpp.input`

and should reproduce results for Supplementary Table S3

The results for this and other example runs with inputs from `ranking_phase_fields/test_input/`
are copied in `ranking_phase_fields/test_results/`.

## Functionality
Unsupervised pattern recognition methods utilise PyOD library

Models:

    'AE'             : AutoEncoder
    'VAE'            : Variational AE
    'ABOD'           : Angle-based outlier detection
    'FeatureBagging' : FeatureBagging
    'HBOS'           : Histogram-Based Outlier Score
    'IForest'        : Isolation Forest
    'KNN'            : K-Nearest Neighbours
    'LOF'            : Local Outlier Factor
    'OCSVM'          : OCSupport Vector Machine
    'PCA'            : Principle Component Analysis
    'SOS'            : SOS(),
    'COF'            : Connectivity-based OF(),
    'CBLOF'          : Clustering-Based LOF
    'SOD'            : SOD(),
    'LOCI'           : LOCI(),
    'MCD'            : MCD()

Read more about these methods https://www.pyod.readthedocs.io

[1] Zhao, Y., Nasrullah, Z. and Li, Z.,
PyOD: A Python Toolbox for Scalable Outlier Detection. 
Journal of machine learning research 20(96), pp.1-7 (2019).

Supported elemental features:

TABLES/AtomicVolume.table
TABLES/AtomicWeight.table
TABLES/BCCbandgap.table
TABLES/BCCefflatcnt.table
TABLES/BCCenergy_pa.table
TABLES/BCCenergydiff.table
TABLES/BCCfermi.table
TABLES/BCCmagmom.table
TABLES/BCCvolume_pa.table
TABLES/BCCvolume_padiff.table
TABLES/BoilingT.table
TABLES/Column.table
TABLES/CovalentRadius.table
TABLES/Density.table
TABLES/Electronegativity.table
TABLES/FirstIonizationEnergy.table
TABLES/GSbandgap.table
TABLES/GSefflatcnt.table
TABLES/GSenergy_pa.table
TABLES/GSestBCClatcnt.table
TABLES/GSestFCClatcnt.table
TABLES/GSmagmom.table
TABLES/GSvolume_pa.table
TABLES/ICSDVolume.table
TABLES/MeltingT.table
TABLES/MendeleevNumber.table
TABLES/MiracleRadius.table
TABLES/NUnfilled.table
TABLES/NValance.table
TABLES/NdUnfilled.table
TABLES/NdValence.table
TABLES/NfUnfilled.table
TABLES/NfValence.table
TABLES/NpUnfilled.table
TABLES/NpValence.table
TABLES/NsUnfilled.table
TABLES/NsValence.table
TABLES/Number.table
TABLES/Pettifor.table
TABLES/Polarizability.table
TABLES/Row.table

[2] Jha, D., Ward, L., Paul, A. et al. 
ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition.
Sci Rep 8, 17593 (2018). https://doi.org/10.1038/s41598-018-35934-y

[3] Glawe, H., Sanna, A., Gross, E. K. U., Marques, M. A. L.,
The optimal one dimensional periodic table: a modified Pettifor chemical scale from data mining.
N. J. Phys. 18, 093011 (2016). https://doi.org/10.1088/1367-2630/18/9/093011

## Parameters of the input file 
(default values are in the rpp.input file)

*icsd_file*    : (default: icsd2017) ICSD excerpt. A text file, a list of ICSD .cif files 
                with specified oxidation states for each element.
*phase_fields*  : (default: quaternary) binary, ternary, quaternary - are supported. 
                Type of phase fields to investigate.
*cations_train* : (default: all) list of elements constituting a phase field in ICSD. 
                Elements in the first positions (cations), e.g. elements for M and M' in MM'AA' phase fields.
*anions_train*  : (default: S,O,Cl,Br,F,N,Te,P,Se,As,I) list of elements constituting a phase field in ICSD. 
                Elements in the last positions (anions) e.g. elements for A and A' in MM'AA' phase fields. 
*nanions_train* : (default: 2) Number of anions (elements with negative oxidation states as specified in icsd_file)
                in the training set. Supported values: 0, 1, 2. If 0 - oxidation states are not taken into account.
*cation1_test*  : (default: Li) list of elements for the first position (e.g. M in MM'AA' phase fields)
                in the phase fields to explore (no reported associated compositions in ICSD). 
*cation2_test*  : (default: all) list of elements for the second position (e.g. M' in MM'AA' phase fields)
                in the phase fields to explore (no reported associated compositions in ICSD).
                Ignored for binary and ternary (type MAA') phase fields.
*anion1_test*   : (default: S,O,Cl,Br,I,F,N) list of elements for the 3rd position (e.g. A in MM'AA' phase fields)
                in the phase fields to explore (no reported associated compositions in ICSD).
                Stands for a 3rd cation in ternary (type MM'M") and quaternary (type MM'M"A) phase fields.
*anion2_test*   : (default: S,O,Cl,Br,I,F,N) list of elements for the 4th position (e.g. A' in MM'AA' phase fields)
                in the phase fields to explore (no reported associated compositions in ICSD).
                Ignored for ternary (type MM'A) phase fields.
*method*        : (default: VAE) See all supported models above.

*cross-validate*: (default: Fault) If True sets 5-fold cross-validation of the model.

*average_runs*  : (default: 1) Number of runs to average the scores over. Makes sense for not neural network based (AE, VAE)
                methods.
*features*      : (default: See rpp.input). See all supported features above.
