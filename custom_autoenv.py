import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms  # Убедитесь, что импортируете transforms
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
learning_rate = 0.001
num_epochs = 100
max_samples = 50000  # Ограничиваем количество изображений

# Загрузка данных
train_dataset = ImageVectorDataset('data/train/images', 'data/train/vectors', max_samples=max_samples)

# Проверка количества изображений
print(f"Общее количество изображений в наборе: {len(train_dataset)}")
if len(train_dataset) == 0:
    raise ValueError("Набор данных пуст. Проверьте наличие изображений в папке.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Пример получения первой партии данных
try:
    for images, vectors in train_loader:
        print(f"Размер батча: {images.size()}")  # Отобразим размер загруженных изображений
        break  # Выход из цикла после первой итерации
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")

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

# Создание папки для сохранения визуализаций
visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

# Список для хранения потерь
losses = []

# Обучение модели
for epoch in range(num_epochs):
    epoch_loss = 0  # Сумма потерь за эпоху
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

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)  # Средняя потеря за эпоху
    losses.append(average_loss)  # Сохраняем среднюю потерю
    print(f"Эпоха [{epoch + 1}/{num_epochs}], Потеря: {average_loss:.4f}")

    # Проверка качества восстановления после каждой эпохи
    if (epoch + 1) % 5 == 0:  # Каждые 5 эпох
        with torch.no_grad():
            test_vectors = vectors[:5].to(device)  # Получаем первые 5 векторов
            reconstructed_images = model(test_vectors)

            # Визуализация
            fig, ax = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(5):
                ax[0, i].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
                ax[0, i].set_title("Оригинал")
                ax[0, i].axis('off')

                ax[1, i].imshow(reconstructed_images[i].cpu().numpy().transpose(1, 2, 0))
                ax[1, i].set_title("Восстановленное")
                ax[1, i].axis('off')

            # Сохранение визуализации
            plt.savefig(os.path.join(visualization_dir, f"epoch_{epoch + 1}.png"))  # Сохраняем изображение
            plt.close()  # Закрываем фигуру

print("Обучение завершено.")

# Сохранение модели
torch.save(model.state_dict(), 'autoencoder.pth')

# Проверка качества восстановления (можно сделать на тестовом наборе)
model.eval()
with torch.no_grad():
    # Загрузите векторы тестовой выборки и проверьте восстановление изображений
    test_dataset = ImageVectorDataset('data/test/images', 'data/test/vectors', max_samples=max_samples)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for images, vectors in test_loader:
        vectors = vectors.to(device)
        reconstructed_images = model(vectors)

        # Здесь вы можете сравнить reconstructed_images с оригинальными images
