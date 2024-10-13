import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import ImageVectorDataset
from model import AdvancedImageRestorationModel
from torch.utils.data import DataLoader


# Настройка устройства (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Параметры
batch_size = 32
learning_rate = 0.0001
num_epochs = 75
latent_vector_size = 512  # Размер вектора от модели ResNet50
max_samples = 25000  # Ограничиваем количество изображений

# Загрузка данных
train_dataset = ImageVectorDataset('data/train/images', 'data/train/vectors', max_samples=max_samples)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
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
    torch.save(model.state_dict(), 'models/image_restoration_model.pth')