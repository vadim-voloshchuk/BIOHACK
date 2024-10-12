import streamlit as st
import numpy as np
import json
import os
import torch
import torch.nn as nn
import onnxruntime as ort
import io
from src.generate import decode_vector_to_image
from PIL import Image
import torchvision.transforms as transforms

# Инициализация ONNX-сессии
weights_path = "models/w600k_r50.onnx"
session = ort.InferenceSession(weights_path)

# Настройка устройства (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Инициализация модели и загрузка весов
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth'))
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
            vector = np.load(uploaded_file, allow_pickle=False).astype(np.float32)  # Убедитесь, что файл корректен
        elif uploaded_file.name.endswith('.json'):
            vector = np.array(json.load(uploaded_file))

        st.write("Полученный вектор:")
        st.write(vector)

        print(vector.shape)
        if st.button("Восстановить изображение"):
            if vector.shape[0] != 512:
                st.error(f"Ожидался вектор размера 512, получен размер {vector.shape[0]}.")
            else:
                vector_tensor = torch.tensor(vector).float().to(device).unsqueeze(0)  # Добавляем размерность batch
                with torch.no_grad():
                    reconstructed_image = model(vector_tensor)

                # Проверяем размерность выходного изображения
                if reconstructed_image.shape[1:] != (3, 112, 112):
                    st.error(f"Некорректная форма выходного изображения: {reconstructed_image.shape[1:]}.")
                else:
                    reconstructed_image = (reconstructed_image.cpu().numpy() * 0.5 + 0.5) * 255  # Обратно в диапазон [0, 255]
                    reconstructed_image = reconstructed_image.astype(np.uint8)
                    image = Image.fromarray(reconstructed_image.reshape(112, 112, 3))
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
