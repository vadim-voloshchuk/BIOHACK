import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Кастомный датасет
class CustomDataset(Dataset):
    def __init__(self, images_dir, vectors_dir, transform=None):
        self.images_dir = images_dir
        self.vectors_dir = vectors_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        vector_path = os.path.join(self.vectors_dir, img_name[:-4] + ".npy")

        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Загружаем вектор
        vector = np.load(vector_path).astype(np.float32)

        # Печать размера вектора для отладки
        print(f"Loaded vector size: {vector.shape}")

        return vector, image

# Определение автокодировщика
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, latent_dim),  # Убедитесь, что вектор имеет размер 512
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),  # Восстанавливаем размер до 512
            nn.ReLU(),
            nn.Unflatten(1, (512,)),  # Измените на (512,) перед `ConvTranspose2d`
            nn.ConvTranspose2d(1, 32, kernel_size=4, stride=2, padding=1),  # Измените количество входных каналов на 1
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Возвращаем к 3 каналам
            nn.Sigmoid()  # Нормализация
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Параметры
latent_dim = 128  # Размер вектора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Загрузка данных
train_images_dir = "data/train/images"
train_vectors_dir = "data/train/vectors"
train_dataset = CustomDataset(train_images_dir, train_vectors_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Инициализация модели, потерь и оптимизатора
model = Autoencoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
num_epochs = 50
for epoch in range(num_epochs):
    for vectors, images in train_loader:
        vectors, images = vectors.to(device), images.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(vectors)

        # Вычисление потерь
        loss = criterion(outputs, images)
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Сохранение модели
torch.save(model.state_dict(), 'autoencoder.pth')
