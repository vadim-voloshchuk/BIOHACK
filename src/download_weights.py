import requests
import os

def download_weights(url, save_path):
    """
    Скачивает веса модели по указанному URL и сохраняет их в указанный файл.

    Args:
        url: URL для скачивания весов.
        save_path: Путь для сохранения весов.
    """
  
    if os.path.exists(save_path):
        print(f"Файл с весами уже существует: {save_path}")
        return

    print(f"Скачивание весов модели из {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192): 
            f.write(chunk)

    print(f"Веса модели успешно скачаны и сохранены в {save_path}")

if __name__ == "__main__":
    weights_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"  # URL для скачивания весов ResNet50@WebFace600K
    save_path = "models/buffalo_l.zip"  # Путь для сохранения весов

    download_weights(weights_url, save_path)