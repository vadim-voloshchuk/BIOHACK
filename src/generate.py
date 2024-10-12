import torch
import torch.nn as nn
from src.core.VAE import VAE

def decode_vector_to_image(vector):
    # Проверка устройства (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Инициализация и загрузка модели
    vae = VAE()
    vae.load_model('vae_model.pth')  # Укажите путь к вашей модели
    vae.eval()
    
    # Перемещение модели на правильное устройство
    vae.to(device)

    # Преобразование вектора в тензор и перемещение на устройство
    vector_tensor = torch.tensor(vector, device=device).unsqueeze(0)  # Добавляем размерность для батча

    with torch.no_grad():
        decoded_vector = vae.decode(vector_tensor).cpu().numpy()  # Декодирование и возврат на CPU
    
    return decoded_vector
