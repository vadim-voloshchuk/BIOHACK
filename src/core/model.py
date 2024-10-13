# Определение архитектуры модели восстановления

import torch.nn as nn

class AdvancedImageRestorationModel(nn.Module):
    def __init__(self):
        super(AdvancedImageRestorationModel, self).__init__()
        
        # Линейная часть для преобразования вектора в "начальное" изображение малого размера
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
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
