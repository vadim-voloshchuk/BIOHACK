
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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