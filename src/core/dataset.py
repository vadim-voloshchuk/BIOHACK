import os
import torch
from torch.utils.data import Dataset
import numpy as np

# Загрузчик данных для embedding-векторов
class FaceVectorDataset(Dataset):
    def __init__(self, vectors_dir):
        self.vectors = []
        for file in os.listdir(vectors_dir):
            if file.endswith(".npy"):
                vector = np.load(os.path.join(vectors_dir, file))
                self.vectors.append(vector)
        self.vectors = np.array(self.vectors, dtype=np.float32)
    
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return torch.tensor(self.vectors[idx])
