# Detekcija dronova putem RF signala

Projekat iz predmeta Tehnologije i alati u mašinskom učenju. Cilj je istrenirati model koji na osnovu radio-frekvencijskog (RF) signala prepoznaje tip drona i razlikuje ga od šuma, uz kompletan MLOps tok: treniranje → AutoML → evaluacija → REST API serviranje.

Prezentacija projekta: [`docs/Detekcija_dronova.pdf`](docs/Detekcija_dronova.pdf).

## Sadržaj

- [Dataset](#dataset)
- [Struktura projekta](#struktura-projekta)
- [Instalacija](#instalacija)
- [Pokretanje](#pokretanje)
- [MLOps komponente](#mlops-komponente)
- [Rezultati](#rezultati)

## Dataset

[Noisy Drone RF Signal Classification](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification) (Kaggle) — 98.705 označenih uzoraka, 7 klasa (6 tipova dronova + Noise), u obliku sirovih IQ signala i gotovih spektrograma.

| Klasa | Broj uzoraka |
|---|---|
| DJI | 2.194 |
| FutabaT14 | 6.938 |
| FutabaT7 | 3.661 |
| Graupner | 6.481 |
| Noise | 52.552 |
| Taranis | 16.546 |
| Turnigy | 10.333 |

`dataset.pt` **nije uključen u ovaj repozitorijum** (fajl je prevelik za Git). Preuzmi ga sa Kaggle-a i stavi u `data/dataset.pt`. Ako je negde drugde, postavi promenljivu okruženja `DRONE_DATASET`:

```powershell
$env:DRONE_DATASET = "D:\podaci\dataset.pt"     # Windows PowerShell
```

```bash
export DRONE_DATASET=/putanja/do/dataset.pt     # Linux/Mac
```

## Struktura projekta

```
├── notebooks/
│   ├── 01_explore_dataset.ipynb    # istraživanje dataset-a, stratifikovan split (70/15/15, seed 42)
│   ├── 02_data_pipeline.ipynb      # PyTorch Dataset/DataLoader
│   ├── 03_baseline_model.ipynb     # ručno dizajniran CNN, trening, evaluacija
│   ├── 04_automl.ipynb             # Optuna pretraga hiperparametara
│   └── 05_evaluation.ipynb         # poređenje baseline vs AutoML (validacioni i test skup)
├── src/
│   ├── config.py                   # sve putanje projekta na jednom mestu
│   ├── drone_dataset.py            # PyTorch Dataset klasa + učitavanje podataka
│   ├── train_utils.py              # trening petlja sa early stopping-om
│   ├── models.py                   # ConfigurableCNN arhitektura
│   ├── api.py                      # FastAPI REST server
│   └── test_api.py                 # testna skripta za API
├── weights/                        # baseline_model.pth, automl_model.pth, automl_best_params.json
├── results/                        # grafici i tabela poređenja modela
├── data/                           # class_stats.csv, SNR_stats.csv, train_val_test_split.npz
├── docs/                           # prezentacija
└── requirements.txt
```

## Instalacija

```bash
git clone https://github.com/Bastaja47/drone-rf-detection.git
cd drone-rf-detection

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Pokretanje

Notebook-ovi se pokreću redom iz foldera `notebooks/` (svaki zavisi od fajlova koje prethodni sačuva):

1. `01_explore_dataset.ipynb` — pravi `data/train_val_test_split.npz`
2. `02_data_pipeline.ipynb` — Dataset/DataLoader provera
3. `03_baseline_model.ipynb` — trenira i čuva `weights/baseline_model.pth`
4. `04_automl.ipynb` — Optuna pretraga, čuva `weights/automl_model.pth` i `weights/automl_best_params.json`
5. `05_evaluation.ipynb` — poređenje modela

Zatim, za REST API (iz korenskog foldera projekta):

```bash
uvicorn api:app --app-dir src --reload
```

U drugom terminalu (dok server radi):

```bash
python src/test_api.py
```

API dokumentacija (interaktivna): `http://127.0.0.1:8000/docs`

API prima spektrogram oblika `[2, 128, 128]` i vraća predviđenu klasu, confidence i verovatnoće za svih 7 klasa. Za drugačiji oblik ulaza vraća grešku 422.

## MLOps komponente

| Komponenta | Status |
|---|---|
| Treniranje i evaluacija (uz AutoML) | Praktično implementirano |
| Primena i serviranje modela (REST API) | Praktično implementirano |
| Monitoring | Teorijski pokriveno |

## Rezultati

Poređenje na **validacionom skupu** (na njemu su birani i hiperparametri i epoha early stopping-a):

| Metrika | Baseline | AutoML |
|---|---|---|
| Accuracy | 93.16% | **94.06%** |
| Macro F1 | 88.94% | **90.74%** |
| Macro Recall | 84.21% | **87.21%** |
| DJI Recall (najteža klasa) | 69.00% | **77.81%** |

AutoML model (Optuna, 20 trial-ova) je pronašao arhitekturu sa 4 konvoluciona sloja, 32 bazna filtera, dropout 0.222, learning rate 0.0007. Poboljšanje u odnosu na baseline je najizraženije kod retkih klasa i pri niskom SNR-u (zašumljeni signali) — detaljna analiza u `notebooks/05_evaluation.ipynb`.

**Gde model greši:** skoro sve greške idu ka klasi Noise, što je posledica niskog SNR-a (signal prekriven šumom). Klase FutabaT7 i FutabaT14 se mešaju međusobno (isti proizvođač, sličan protokol).

**Test API-ja:** na po jednom uzorku svake klase iz test skupa pogođeno je 6 od 7 (greška: FutabaT7 predviđen kao FutabaT14, sa confidence-om 0.67).

Na kraju `05_evaluation.ipynb` nalazi se i evaluacija oba modela na **test skupu**, koji nije korišćen ni za jedan izbor (`results/final_comparison_test.csv`).

## Autor

Stojan Bastaja, projekat izrađen u okviru predmeta Tehnologije i alati u mašinskom učenju.
