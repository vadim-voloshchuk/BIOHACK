import torch
import torch.nn as nn
import numpy as np
from src.core.VAE import VAE

def decode_vector_to_image(vector):
    vector = np.asarray(vector)
    if vector.shape[0] != 512:  # Убедитесь, что размер 512
        raise ValueError(f"Expected vector of size 512, got {vector.shape[0]}")

    vae = VAE()
    vae.load_model('vae_model.pth')  # Укажите путь к вашей модели
    vae.eval()

    # Преобразование вектора в тензор и передача в декодер
    vector_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)  # Добавление batch dimension
    with torch.no_grad():
        decoded_vector = vae.decode(vector_tensor).cpu().numpy()

    return decoded_vector