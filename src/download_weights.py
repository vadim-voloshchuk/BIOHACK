import requests
import os
import zipfile

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
    unzip_weights(save_path)

def unzip_weights(zip_path):
    """
    Распаковывает архив с весами в ту же директорию, где находится архив.

    Args:
        zip_path: Путь до архива с весами.
    """
    extract_dir = os.path.dirname(zip_path)
    print(f"Распаковка архива {zip_path} в {extract_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"Архив успешно распакован в {extract_dir}")

if __name__ == "__main__":
    weights_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"  # URL для скачивания весов ResNet50@WebFace600K
    save_path = "models/buffalo_l.zip"  # Путь для сохранения весов

    download_weights(weights_url, save_path)
