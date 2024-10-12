import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from VAE import VAE  # Импорт архитектуры VAE
from dataset import FaceVectorDataset  # Импорт класса Dataset

# Пути к обучающим и тестовым векторам
train_vectors_dir = "data/train/vectors"
test_vectors_dir = "data/test/vectors"

# Параметры обучения
input_dim = 512  # Размер embedding-векторов
latent_dim = 128  # Размер скрытого пространства
num_epochs = 50
learning_rate = 1e-3

# Создание датасетов и загрузчиков данных
train_dataset = FaceVectorDataset(train_vectors_dir)
test_dataset = FaceVectorDataset(test_vectors_dir)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Инициализация модели, оптимизатора
vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Функция обучения
def train_vae(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = model.loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader.dataset)}")

# Запуск обучения
train_vae(vae, train_loader, optimizer, num_epochs)

# Сохранение модели и оптимизатора
torch.save({
    'epoch': num_epochs,
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "vae_model.pth")
