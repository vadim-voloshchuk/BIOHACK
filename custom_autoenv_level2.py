import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
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
num_epochs = 50
latent_vector_size = 64  # Размер скрытого вектора
max_samples = 5000  # Ограничиваем количество изображений

# Загрузка данных
train_dataset = ImageVectorDataset('data/train/images', 'data/train/vectors', max_samples=max_samples)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Определение архитектуры генератора
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_vector_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 112 * 112),  # Генерируем изображение размером 112x112
            nn.Tanh()  # Используем Tanh для нормализации выходных данных
        )

    def forward(self, x):
        return self.model(x).view(-1, 3, 112, 112)

# Определение архитектуры дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 112 * 112, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),  # Выход - вероятность
            nn.Sigmoid()  # Используем Sigmoid для вероятности
        )

    def forward(self, x):
        return self.model(x.view(-1, 3 * 112 * 112))

# Инициализация моделей, функции потерь и оптимизаторов
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Создание папки для сохранения визуализаций
visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

# Обучение GAN
for epoch in range(num_epochs):
    for i, (images, vectors) in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.size(0)

        # Обучение дискриминатора
        optimizer_d.zero_grad()

        # Реальные изображения
        real_labels = torch.ones(batch_size, 1).to(device)
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)

        # Генерация фейковых изображений
        noise = torch.randn(batch_size, latent_vector_size).to(device)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        outputs = discriminator(fake_images.detach())  # Не вычисляем градиенты для генератора
        d_loss_fake = criterion(outputs, fake_labels)

        # Общая потеря дискриминатора
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Обучение генератора
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # Генератор хочет, чтобы дискриминатор думал, что изображения реальные
        g_loss.backward()
        optimizer_g.step()

        if (i + 1) % 100 == 0:  # Печатаем каждые 100 шагов
            print(f"Эпоха [{epoch + 1}/{num_epochs}], Шаг [{i + 1}/{len(train_loader)}], Потеря D: {d_loss.item():.4f}, Потеря G: {g_loss.item():.4f}")

    # Сохранение визуализации
    with torch.no_grad():
        test_noise = torch.randn(10, latent_vector_size).to(device)
        generated_images = generator(test_noise)

        # Визуализация
        fig, ax = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            ax[0, i].imshow(generated_images[i].cpu().numpy().transpose(1, 2, 0))
            ax[0, i].set_title("Сгенерированное")
            ax[0, i].axis('off')

            ax[1, i].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
            ax[1, i].set_title("Оригинал")
            ax[1, i].axis('off')

        plt.savefig(os.path.join(visualization_dir, f"epoch_{epoch + 1}.png"))  # Сохраняем изображение
        plt.close()  # Закрываем фигуру

print("Обучение завершено.")

# Сохранение моделей
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# Проверка качества генерации (можно сделать на тестовом наборе)
generator.eval()
with torch.no_grad():
    # Генерация изображений с помощью генератора
    test_noise = torch.randn(5, latent_vector_size).to(device)
    generated_images = generator(test_noise)

    # Визуализация
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax[i].imshow(generated_images[i].cpu().numpy().transpose(1, 2, 0))
        ax[i].set_title("Сгенерированное")
        ax[i].axis('off')

    plt.show()
