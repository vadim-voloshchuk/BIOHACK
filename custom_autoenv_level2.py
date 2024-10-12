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

        return vector, image  # Возвращаем вектор и соответствующее изображение

# Параметры
batch_size = 32
learning_rate = 0.0001
num_epochs = 50
latent_vector_size = 512  # Размер вектора от модели ResNet50
max_samples = 5000  # Ограничиваем количество изображений

# Загрузка данных
train_dataset = ImageVectorDataset('data/train/images', 'data/train/vectors', max_samples=max_samples)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Определение архитектуры модели восстановления
class AdvancedImageRestorationModel(nn.Module):
    def __init__(self):
        super(AdvancedImageRestorationModel, self).__init__()
        
        # Линейная часть для преобразования вектора в "начальное" изображение малого размера
        self.fc = nn.Sequential(
            nn.Linear(latent_vector_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256 * 7 * 7),  # Преобразуем в изображение размера 7x7 с 256 каналами
            nn.ReLU()
        )

        # Блоки для прогрессивного увеличения изображения с использованием сверточных слоев
        self.deconv_blocks = nn.Sequential(
            # Первый блок: увеличиваем изображение до 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Второй блок: увеличиваем изображение до 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Третий блок: увеличиваем изображение до 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Четвертый блок: увеличиваем изображение до 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Последний слой: 3 канала для изображения RGB
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),  # 112x112, 3 канала (RGB)
            nn.Tanh()  # Используем Tanh для нормализации значений в диапазон [-1, 1]
        )

    def forward(self, x):
        # Пропускаем через полносвязную сеть и изменяем форму на 256 каналов с размером 7x7
        x = self.fc(x).view(-1, 256, 7, 7)
        
        # Пропускаем через сверточные транспонированные слои для увеличения изображения
        x = self.deconv_blocks(x)
        return x
    
# Инициализация модели, функции потерь и оптимизатора
model = AdvancedImageRestorationModel().to(device)
criterion = nn.MSELoss()  # Используем MSE для сравнения восстанавливаемых и реальных изображений
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Создание папки для сохранения визуализаций
visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

# Обучение модели восстановления
for epoch in range(num_epochs):
    for i, (vectors, images) in enumerate(train_loader):
        vectors = vectors.to(device)
        images = images.to(device)
        batch_size = images.size(0)

        # Прямой проход
        optimizer.zero_grad()
        outputs = model(vectors)
        loss = criterion(outputs, images)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:  # Печатаем каждые 100 шагов
            print(f"Эпоха [{epoch + 1}/{num_epochs}], Шаг [{i + 1}/{len(train_loader)}], Потеря: {loss.item():.4f}")

    # Сохранение визуализации
    with torch.no_grad():
        test_vectors, test_images = next(iter(train_loader))
        test_vectors = test_vectors.to(device)
        generated_images = model(test_vectors)

        # Визуализация
        fig, ax = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            ax[0, i].imshow(generated_images[i].cpu().numpy().transpose(1, 2, 0))
            ax[0, i].set_title("Восстановленное")
            ax[0, i].axis('off')

            ax[1, i].imshow(test_images[i].cpu().numpy().transpose(1, 2, 0))
            ax[1, i].set_title("Оригинал")
            ax[1, i].axis('off')

        plt.savefig(os.path.join(visualization_dir, f"epoch_{epoch + 1}.png"))
        plt.close()

print("Обучение завершено.")

# Сохранение модели
torch.save(model.state_dict(), 'image_restoration_model.pth')

# Проверка качества восстановления
model.eval()
with torch.no_grad():
    test_vectors, test_images = next(iter(train_loader))
    test_vectors = test_vectors.to(device)
    generated_images = model(test_vectors)

    # Визуализация
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax[i].imshow(generated_images[i].cpu().numpy().transpose(1, 2, 0))
        ax[i].set_title("Восстановленное")
        ax[i].axis('off')

    plt.show()
