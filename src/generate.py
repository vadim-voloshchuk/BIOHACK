import torch
import torch.nn as nn
from core.VAE import VAE

def decode_vector_to_image(vector):
    vae = VAE()
    vae.load_model('vae_model.pth')  # Укажите путь к вашей модели
    vae.eval()
    with torch.no_grad():
        decoded_vector = vae.decode(torch.tensor(vector)).numpy()
    return decoded_vector
