# Alzheimer MRI CNN 

This project implements a 3D convolutional neural network (CNN) for Alzheimer’s disease classification from MRI scans (OASIS) and focuses on patient-level analysis, uncertainty estimation, and qualitative inspection of model failures.

The project supports:

- **MRI preprocessing pipeline** (OASIS cross-sectional)
- **3D CNN training and evaluation** with stratified k-fold cross-validation
- **Patient-level prediction aggregation** (test-time augmentation)
- **Relational database** (SQLite) for metadata, features, and predictions
- **Exploratory data analysis (EDA) and uncertainty analysis** via Jupyter notebooks
- **Qualitative inspection of failure cases** through MRI slice visualization

## Project structure
```
alzheimer_mri_cnn/
│
├── src/                        # contains all the python scripts
│   ├── data_download.py        # downloads OASIS data into data/raw folder
│   ├── neural_networks.py      # available neural networks models
│   ├── oasis_dataset.py        # dataset class and related functions
│   ├── preprocessing.py        # process raw data into .npy volumes, saving meta-data into .json datasets
│   ├── train_evaluate.py       # main script
│   └── util.py                 # helper functions  
├── data/
│   ├── raw/                    # tmp folder with downloaded OASIS zips (ignored)
│   └── processed/              # folder with final processed volumes 
│       └── oasis.db            # patient-level SQLite database
│
├── inputs/                     # contains input files
│
├── notebooks/
│   ├── 01_eda_global_features.ipynb
│   ├── 02_model_confidence_uncertainty.ipynb
│   └── 03_failure_cases_qualitative_analysis.ipynb
│
├── outputs/                    # (ignored)
│   └── figures/
│
├── requirements.txt
└── README.md
```

## Database Design

The SQLite database (oasis.db) contains:

- patient metadata
- volumetric descriptors
- ground-truth labels
- model predictions and confidence scores

This design decouples model training from data analysis, allowing notebooks to operate purely on meta-data.

## Notebooks Overview

### Exploratory Data Analysis
- Global volumetric feature distributions
- Class-wise comparisons
- Dataset sanity checks

### Model Confidence & Uncertainty
- Patient-level confidence distributions
- Uncertainty vs correctness
- Identification of ambiguous cases

### Qualitative Failure Analysis
- Visual inspection of true positives, false positives, true negatives and false negatives
- Slice-based MRI visualization
- Interpretation of model errors

## Requirements

- Python 3.10+
- pip
- (optional) GPU with CUDA support for training

Install dependencies:
```
pip install -r requirements.txt
```

## Training and Evaluation Configuration

The training and evaluation pipeline is controlled via command-line arguments passed to train_evaluate.py. These options allow flexible configuration of the model architecture, data preprocessing and augmentation strategy.
For reproducibility and clarity, configurations can be provided either directly via command-line flags or through an input configuration file.

### Network Architecture
- -n, --net_type: Specifies the type of neural network to use (e.g. CNN variants).  If not provided, a default architecture is selected. For a list of architectures, check neural_network.py
### Data Preprocessing and Augmentation
- -t, --transform: Enables deterministic preprocessing transformations applied to the training data (e.g. normalization, resizing). This option affects training data only.
- -a, --augment <int>: Performs class-balancing data augmentation by applying random transformations to the minority class. The integer value specifies the number of augmented samples generated per original sample.
- -s, --sampler: Enables a sampling-based strategy to balance class distributions during training. This option provides an alternative to explicit data augmentation.
Only one of --augment or --sampler should be used at a time.
### Evaluation-Time Augmentation
- -at, --aug_test: Enables test-time augmentation (TTA) during evaluation. Multiple transformed views of the same subject are generated, and predicted probabilities are averaged at patient level.
This option is intended exclusively for uncertainty estimation and qualitative analysis, not for performance comparison.
### Cross-Validation
- -k, --kfolds <int>: Enables stratified k-fold cross-validation, where the integer value specifies the number of folds. For each fold: the model is trained on the training split, evaluated on the corresponding evaluation split,
  patient-level predictions are stored in the database. No independent test set is used.
### Dataset Configuration
- -d, --discs <int>: Specifies the number of OASIS discs to download and preprocess. This option is intended for controlled experiments or reduced-scale testing.
### Input Configuration File
- -i, --input <path>: Reads all configuration options from an external input file. This mode is recommended for reproducibility and experiment tracking.
  
## Recommended Usage

For exploratory experiments, command-line flags can be used directly.
For reproducible experiments and final analyses, using an input configuration file is strongly recommended.

Example:
```
python src/train_evaluate.py --input 19_12_25_1923
```

Inputs file should have the .txt extension and should be placed in the inputs folder. Only the name, without the file extension, should be provided, not the path. For examples, take a look at the files in the inputs folder.

## Notes

- Training and evaluation are performed per fold when k-fold cross-validation is enabled.
- Stored predictions, rather than trained models, constitute the primary output of the pipeline. All downstream analyses operate exclusively on the generated SQLite database.
- Raw MRI data and large intermediate files are excluded from version control
- Notebooks are deterministic given the database state
