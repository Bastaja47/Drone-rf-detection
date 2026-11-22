"""
test_api.py

Test skripta - uzima STVARNE uzorke iz TEST skupa (model ih nikad nije video)
i šalje ih API-ju, da proverimo ponašanje na raznovrsnim, nepristrasnim primerima.

Pokretanje (dok server radi u drugom terminalu):  python src/test_api.py
"""

import requests
import torch
import numpy as np

from config import DATASET_PATH, SPLIT_PATH

API_URL = "http://127.0.0.1:8000/predict"

data = torch.load(DATASET_PATH, weights_only=False)
split = np.load(SPLIT_PATH)
test_indices = split['test_indices']

class_names = ['DJI', 'FutabaT14', 'FutabaT7', 'Graupner', 'Noise', 'Taranis', 'Turnigy']

# Zaglavlje tabele
print(f"{'Stvarna klasa':<12} {'Predikcija':<12} {'Confidence':<12} {'SNR':<6} {'Tačno?'}")
print("-" * 60)

for class_id in range(7):
    # Iz test skupa biramo uzorke ove klase
    class_mask = data['y'][test_indices] == class_id
    class_test_indices = test_indices[class_mask.numpy()]

    # Uzimamo prvi test uzorak te klase
    sample_idx = class_test_indices[0]

    spectrogram = data['x_spec'][sample_idx]
    true_label = data['y'][sample_idx].item()
    snr = data['snr'][sample_idx].item()

    # JSON ne zna za tenzore, pa spektrogram saljemo kao ugnjezdenu listu
    spectrogram_list = spectrogram.tolist()

    response = requests.post(API_URL, json={"data": spectrogram_list})

    result = response.json()
    predicted = result['predicted_class']
    confidence = result['confidence']
    correct = "✓" if predicted == class_names[true_label] else "✗"

    print(f"{class_names[true_label]:<12} {predicted:<12} {confidence:<12} {snr:<6} {correct}")
