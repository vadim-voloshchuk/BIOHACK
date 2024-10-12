import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Добавьте эту строку, если она отсутствует


# Настройка устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пользовательский Dataset для загрузки векторов и изображений
class ImageVectorDataset(Dataset):
    def __init__(self, images_dir, vectors_dir):
        self.images_dir = images_dir
        self.vectors_dir = vectors_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        vector_name = os.path.join(self.vectors_dir, self.image_files[idx][:-4] + '.npy')

        image = Image.open(img_name).convert('RGB')
        image = transforms.ToTensor()(image)

        vector = np.load(vector_name).astype(np.float32)

        return image, vector

# Параметры
batch_size = 32
learning_rate = 0.001
num_epochs = 20

# Загрузка данных
train_dataset = ImageVectorDataset('data/train/images', 'data/train/vectors')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Определение архитектуры автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),  # Предположим, что размер вектора 512
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 112 * 112),  # Восстанавливаем изображение размером 112x112
            nn.Tanh()  # Используем Tanh для нормализации выходных данных
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 3, 112, 112)  # Возвращаем форму (batch_size, channels, height, width)

# Инициализация модели, функции потерь и оптимизатора
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(num_epochs):
    for images, vectors in train_loader:
        images = images.to(device)
        vectors = vectors.to(device)

        # Прямой проход
        outputs = model(vectors)
        loss = criterion(outputs, images)

        # Обратное распространение и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Эпоха [{epoch + 1}/{num_epochs}], Потеря: {loss.item():.4f}")

print("Обучение завершено.")

# Сохранение модели
torch.save(model.state_dict(), 'autoencoder.pth')

# Проверка качества восстановления (можно сделать на тестовом наборе)
model.eval()
with torch.no_grad():
    # Загрузите векторы тестовой выборки и проверьте восстановление изображений
    test_dataset = ImageVectorDataset('data/test/images', 'data/test/vectors')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for images, vectors in test_loader:
        vectors = vectors.to(device)
        reconstructed_images = model(vectors)

        # Здесь вы можете сравнить reconstructed_images с оригинальными images
