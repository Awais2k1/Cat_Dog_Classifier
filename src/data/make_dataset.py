import os
import zipfile
from tensorflow.keras.utils import image_dataset_from_directory
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """Download dataset from Kaggle"""
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('salader/dogs-vs-cats', path='data/raw', unzip=True)

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_datasets(data_dir, image_size=(256, 256), batch_size=32):
    """Create train and validation datasets"""
    train_ds = image_dataset_from_directory(
        directory=os.path.join(data_dir, 'train'),
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size
    )
    
    val_ds = image_dataset_from_directory(
        directory=os.path.join(data_dir, 'test'),
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size
    )
    
    return train_ds, val_ds

def prepare_data():
    """Main function to prepare all data"""
    if not os.path.exists('data/raw/dogs-vs-cats.zip'):
        download_dataset()
    
    if not os.path.exists('data/raw/train'):
        extract_zip('data/raw/dogs-vs-cats.zip', 'data/raw')
    
    return create_datasets('data/raw')