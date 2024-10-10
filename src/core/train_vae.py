import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from .VAE import VAE  # Изменение: относительный импорт VAE

# Параметры обучения
epochs = 100
batch_size = 64
learning_rate = 1e-3
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к данным
train_data_dir = "data/train/images" 

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Размер, используемый в VAE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Загрузка данных
train_dataset = ImageFolder(train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Создание модели VAE
vae = VAE(latent_dim=latent_dim).to(device)

# Оптимизатор и функция потерь
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

def loss_function(recon_x, x, mu, logvar):
    """Функция потерь VAE (ELBO)."""
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # Reconstruction loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return BCE + KLD

# Обучение VAE
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch: {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# Сохранение модели
torch.save(vae.state_dict(), "models/vae.pth")

print("Обучение VAE завершено.")