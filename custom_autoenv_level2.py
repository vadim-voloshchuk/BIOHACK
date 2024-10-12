import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Настройка устройства (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageVectorDataset(Dataset):
    def __init__(self, images_dir, vectors_dir, max_samples=None):
        self.images_dir = images_dir
        self.vectors_dir = vectors_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]  # Ограничиваем размер выборки

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        vector_name = os.path.join(self.vectors_dir, self.image_files[idx][:-4] + '.npy')

        image = Image.open(img_name).convert('RGB')
        image = image.resize((112, 112))  # Изменение размера изображения на 112x112
        image = transforms.ToTensor()(image)

        vector = np.load(vector_name).astype(np.float32)

        return image, vector

# Параметры
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
max_samples = 50000  # Ограничиваем количество изображений

# Загрузка данных
train_dataset = ImageVectorDataset('data/train/images', 'data/train/vectors', max_samples=max_samples)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Проверка количества изображений
print(f"Общее количество изображений в наборе: {len(train_dataset)}")
if len(train_dataset) == 0:
    raise ValueError("Набор данных пуст. Проверьте наличие изображений в папке.")

# Определение генератора (автоэнкодера)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8))  # Преобразуем в 64x8x8 для свёрточной обработки
        )
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1),
            nn.Tanh()  # Нормализация выходных данных
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_layers(x)
        return x.view(-1, 3, 112, 112)  # Возвращаем форму (batch_size, channels, height, width)

# Определение дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112 -> 56x56
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56 -> 28x28
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 1),  # Подсчитываем, реальное это или фейковое
            nn.Sigmoid()  # Выход в диапазоне [0, 1]
        )

    def forward(self, x):
        return self.model(x)

# Инициализация моделей
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Оптимизаторы
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Цикл обучения
for epoch in range(num_epochs):
    for images, vectors in train_loader:
        images = images.to(device)
        vectors = vectors.to(device)

        # Обучение дискриминатора
        optimizer_D.zero_grad()
        
        # Генерируем фейковые изображения
        fake_images = generator(vectors)
        
        # Потеря для реальных изображений
        real_labels = torch.ones(images.size(0), 1).to(device)
        real_loss = criterion(discriminator(images), real_labels)

        # Потеря для фейковых изображений
        fake_labels = torch.zeros(images.size(0), 1).to(device)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Обучение генератора
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_images), real_labels)  # Генератор пытается обмануть дискриминатор
        g_loss.backward()
        optimizer_G.step()

    print(f"Эпоха [{epoch + 1}/{num_epochs}], Потеря D: {d_loss.item():.4f}, Потеря G: {g_loss.item():.4f}")

print("Обучение завершено.")
