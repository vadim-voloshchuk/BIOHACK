import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from insightface.app import FaceAnalysis
from .VAE import VAE  # Изменение: относительный импорт VAE
from sklearn.metrics.pairwise import cosine_similarity
import os

# Параметры оценки
batch_size = 64
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.7  # Пороговое значение для косинусного сходства

# Пути к данным и модели
test_data_dir = "data/test"
vae_model_path = "models/vae.pth"
arcface_model_path = "models/buffalo_l.zip"  # Путь к модели ArcFace (ResNet50@WebFace600K)

# Загрузка модели VAE
vae = VAE(latent_dim=latent_dim).to(device)
vae.load_state_dict(torch.load(vae_model_path))
vae.eval()

# Инициализация ArcFace
app = FaceAnalysis(name="buffalo_l", root="models")
app.prepare(ctx_id=0, det_size=(640, 640))

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Размер, используемый в VAE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Загрузка тестовых данных (векторы)
test_vectors = []
test_filenames = []
for filename in os.listdir(os.path.join(test_data_dir, "vectors")):  # Изменение: путь к векторам
    if filename.endswith(".npy"):
      vector = np.load(os.path.join(test_data_dir, "vectors", filename))
      test_vectors.append(vector)
      test_filenames.append(filename[:-4] + ".jpg") 

# Оценка качества восстановления
correct_count = 0
for i, vector in enumerate(test_vectors):
    # Генерация изображения из вектора с помощью VAE
    z = torch.from_numpy(vector).float().unsqueeze(0).to(device)  # Преобразование вектора в тензор
    reconstructed_image = vae.generate(z)

    # Сохранение восстановленного изображения (опционально)
    # save_image(reconstructed_image, f"reconstructed_images/{test_filenames[i]}") 

    # Извлечение вектора из восстановленного изображения с помощью ArcFace
    reconstructed_image_pil = transforms.ToPILImage()(reconstructed_image.squeeze(0))
    faces = app.get(reconstructed_image_pil)
    if faces:
      reconstructed_vector = faces[0].embedding
    else:
      print(f"Не удалось обнаружить лицо на восстановленном изображении: {test_filenames[i]}")
      continue
    
    # Сравнение векторов с помощью косинусного сходства
    similarity = cosine_similarity([vector], [reconstructed_vector])[0][0]
    
    if similarity >= threshold:
        correct_count += 1

accuracy = correct_count / len(test_vectors)
print(f"Accuracy: {accuracy:.4f}")

print("Оценка качества восстановления завершена.") 