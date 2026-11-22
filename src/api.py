"""
api.py

REST API za serviranje drone RF klasifikacionog modela.
Pokreće se iz korenskog foldera projekta:

    uvicorn api:app --app-dir src --reload

Interaktivna dokumentacija: http://127.0.0.1:8000/docs
"""

import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from config import WEIGHTS_DIR
from models import ConfigurableCNN

app = FastAPI(title="Drone RF Detection API")

# Koristi GPU ako postoji, inače CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparametre koje je Optuna pronašla učitavamo da bismo sagradili istu
# arhitekturu pre nego što u nju stavimo sačuvane težine
with open(WEIGHTS_DIR / 'automl_best_params.json', 'r') as f:
    best_params = json.load(f)

model = ConfigurableCNN(
    num_conv_layers=best_params['num_conv_layers'],
    base_filters=best_params['base_filters'],
    dropout_rate=best_params['dropout_rate']
).to(device)
model.load_state_dict(torch.load(WEIGHTS_DIR / 'automl_model.pth', weights_only=True, map_location=device))
model.eval()  # isključuje dropout; model se učitava jednom, pri pokretanju servera

# Redosled mora biti isti kao u dataset-u (indeks klase -> ime)
CLASS_NAMES = ['DJI', 'FutabaT14', 'FutabaT7', 'Graupner', 'Noise', 'Taranis', 'Turnigy']

print(f"Model učitan na uređaj: {device}")


# Pydantic model: FastAPI sam proverava da je ulaz lista listi listi brojeva
class SpectrogramInput(BaseModel):
    """
    Klijent mora poslati 'data' polje: 3D lista brojeva oblika [2, 128, 128]
    (isti oblik kao jedan uzorak iz x_spec u našem dataset-u).
    """
    data: List[List[List[float]]]


@app.get("/")
def root():
    return {"poruka": "Drone RF Detection API radi!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: SpectrogramInput):
    """
    Prima spektrogram [2, 128, 128], vraća predviđenu klasu i verovatnoće
    za sve klase.
    """
    spectrogram = torch.tensor(input_data.data, dtype=torch.float32)

    # Model ocekuje tacno [2, 128, 128]; za drugaciji oblik vracamo gresku klijentu
    if spectrogram.shape != (2, 128, 128):
        raise HTTPException(
            status_code=422,
            detail=f"Ocekivan oblik [2, 128, 128], dobijen {list(spectrogram.shape)}",
        )

    # Dodajemo batch dimenziju: [2, 128, 128] -> [1, 2, 128, 128]
    spectrogram = spectrogram.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(spectrogram)  # logiti, oblik [1, 7]

        # Softmax pretvara logite u verovatnoće koje se sabiraju na 1
        probabilities = F.softmax(outputs, dim=1)[0]

        predicted_idx = torch.argmax(probabilities).item()

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item(), 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(probabilities[i].item(), 4)
            for i in range(len(CLASS_NAMES))
        }
    }
