import streamlit as st
import numpy as np
import json
import os
import torch
import torch.nn as nn
import onnxruntime as ort
import io
from PIL import Image
import torchvision.transforms as transforms

# Инициализация ONNX-сессии
weights_path = "models/w600k_r50.onnx"
session = ort.InferenceSession(weights_path)

# Настройка устройства (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Определение архитектуры модели восстановления
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

# Инициализация модели и загрузка весов
model = AdvancedImageRestorationModel().to(device)
model.load_state_dict(torch.load('image_restoration_model.pth'))
model.eval()  # Переводим модель в режим оценки

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

st.title("VAE Image Processing")

# Страница для восстановления изображения из вектора
page = st.sidebar.selectbox("Выберите действие", ["Восстановить изображение", "Получить вектор из изображения"])


if page == "Восстановить изображение":
    st.header("Восстановление изображения из вектора")

    st.write("Загрузите файл с векторами (формат .npy или .json):")
    uploaded_file = st.file_uploader("Выберите файл", type=["npy", "json"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.npy'):
            vector = np.load(uploaded_file, allow_pickle=False).astype(np.float32)
        elif uploaded_file.name.endswith('.json'):
            vector = np.array(json.load(uploaded_file))

        st.write("Полученный вектор:")
        st.write(vector)

        # Проверка формы вектора
        if vector.shape != (1, 512):
            st.error(f"Ожидался вектор размера (1, 512), получен размер {vector.shape}.")
        else:
            if st.button("Восстановить изображение"):
                vector_tensor = torch.tensor(vector).float().to(device)
                with torch.no_grad():
                    reconstructed_image = model(vector_tensor)

                # Проверяем размерность выходного изображения
                if reconstructed_image.shape != (1, 3, 112, 112):
                    st.error(f"Некорректная форма выходного изображения: {reconstructed_image.shape}.")
                else:
                    # Преобразование обратно в диапазон [0, 255]
                    reconstructed_image = (reconstructed_image.cpu().numpy().squeeze() * 0.5 + 0.5) * 255
                    reconstructed_image = reconstructed_image.astype(np.uint8)

                    # Создание и отображение изображения
                    image = Image.fromarray(reconstructed_image.transpose(1, 2, 0))  # Преобразование в (height, width, channels)
                    st.image(image, caption="Восстановленное изображение", use_column_width=True)

if page == "Получить вектор из изображения":
    st.header("Получение вектора из изображения")
    
    st.write("Загрузите изображение:")
    uploaded_image = st.file_uploader("Выберите изображение", type=["jpg", "png"])

    if uploaded_image is not None:
        # Открытие и преобразование изображения
        img = Image.open(uploaded_image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).numpy()  # Добавление размерности batch

        # Извлечение вектора через ONNX-модель
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        vector = session.run([output_name], {input_name: img_tensor})[0]

        st.write("Извлечённый вектор:")
        st.write(vector)
        st.write("Форма вектора:", vector.shape)  # Вывод формы вектора

        # Сохранение вектора в байтовый поток
        vector_filename = st.text_input("Введите имя файла для сохранения (без расширения):", "")
        if st.button("Скачать вектор"):
            if vector_filename:
                # Создание байтового потока
                buffer = io.BytesIO()
                np.save(buffer, vector)  # Сохранение вектора в байтовый поток
                buffer.seek(0)  # Перемещение указателя на начало потока
                
                st.download_button(label="Скачать вектор в формате .npy",
                                   data=buffer,
                                   file_name=f"{vector_filename}.npy",
                                   mime='application/octet-stream')
                st.success(f"Вектор сохранен как {vector_filename}.npy")
