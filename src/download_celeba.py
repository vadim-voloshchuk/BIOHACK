from kaggle.api.kaggle_api_extended import KaggleApi

def download_celeba_from_kaggle():
    """
    Скачивает датасет CelebA с Kaggle.
    """
    
    # Инициализация Kaggle API
    api = KaggleApi()
    api.authenticate()

    # ID датасета на Kaggle
    dataset_id = "jessicali9530/celeba-dataset" 

    # Скачивание датасета
    api.dataset_download_files(dataset_id, path="data", unzip=True)

    print("Датасет CelebA успешно скачан с Kaggle.")

if __name__ == "__main__":
    download_celeba_from_kaggle()