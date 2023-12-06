"""
Contains various utility functions for PyTorch model training and saving.
"""

import os
import pathlib
import requests
import shutil
import zipfile
import torch
import matplotlib.pyplot as plt
from PIL import Image

def unzip_data(zip_path: pathlib.Path,
              dest_path: pathlib.Path,
              remove_file: bool=False):
  '''Upzip a file from zipref to dest foler, if dest already exists, it will be replaced

  Args:
    zip_path (pathlib.Path): zipfile path
    dest_path (pathlib.Path): destination folder
  '''
  if dest_path.is_dir():
    print(f'[INFO] {dest_path} already exists, removing and recreating')
    shutil.rmtree(dest_path, ignore_errors=True)

  dest_path.mkdir(parents=True, exist_ok=True)

  with zipfile.ZipFile(zip_path) as zip_ref:
    print(f'[INFO] Upzipping {zip_path} into {dest_path}')
    zip_ref.extractall(dest_path)

  if remove_file:
    os.remove(zip_path)


def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"[INFO] Found {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")


def plot_loss_curve(results: dict[str, list[float]]):
  '''
  Plots training curves of a results dictionary

  Args:
    results (dict): dictionary containing list of values,
    {'train_loss': [...],
    'train_acc': [...],
    'test_loss': [...],
    'test_loss': [...]}
  '''
  epochs = range(len(results['train_loss']))
  train_loss = results['train_loss']
  train_acc = results['train_acc']
  test_loss = results['test_loss']
  test_acc = results['test_acc']

  # Setup plot
  plt.figure(figsize=(12, 6))

  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_loss, label='train_loss')
  plt.plot(epochs, test_loss, label='test_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot Accuracy
  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_acc, label='train_acc')
  plt.plot(epochs, test_acc, label='test_acc')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
