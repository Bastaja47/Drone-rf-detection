"""
config.py

Putanje za ceo projekat na jednom mestu.

Skripte u src/ i notebook-ovi u notebooks/ koriste ove putanje umesto
golih imena fajlova, pa rade bez obzira iz kog foldera se pokrenu.

dataset.pt je ogroman i nije u repozitorijumu. Podrazumevano se trazi u
data/dataset.pt, a ako je negde drugde, postavi promenljivu okruzenja
DRONE_DATASET:

    PowerShell:  $env:DRONE_DATASET = "D:\\podaci\\dataset.pt"
    Linux/Mac:   export DRONE_DATASET=/putanja/do/dataset.pt
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
WEIGHTS_DIR = PROJECT_DIR / "weights"
RESULTS_DIR = PROJECT_DIR / "results"

# Ceo dataset sa Kaggle-a (nije u Git-u)
DATASET_PATH = Path(os.environ.get("DRONE_DATASET", DATA_DIR / "dataset.pt"))

# Indeksi uzoraka za train/val/test (pravi ih notebooks/01_explore_dataset.ipynb)
SPLIT_PATH = DATA_DIR / "train_val_test_split.npz"
