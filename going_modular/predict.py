"""
Contains functions for making prediction
"""
import torch
import torchvision
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm.auto import tqdm
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
  def __getitem__(self, index):
    original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
    path = self.imgs[index][0]
    tuple_with_path = (original_tuple + (path,))
    return tuple_with_path

def make_predictions(model: torch.nn.Module,
                     data_path: pathlib.PosixPath,
                     device: torch.device=device):
  '''Make Prediction given a model as model and data as ImageFolder data format

  Args:
    model: torch model for classficiation
    data: datasets.ImageFolder list

  Return:
    Prediction Probabilities in list in device 'cpu'

  '''

  data_transform = transforms.Compose([transforms.Resize(size= (224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  data = ImageFolderWithPaths(root=data_path,
                              transform=data_transform,
                              target_transform=None)

  pred_probs = []
  pred_label = []
  data_path = []
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for sample, y, path in tqdm(data):
      # Map sample into tensor shape for model
      X = torch.unsqueeze(sample, dim=0).to(device)
      # Forward pass -> logits
      y_logits = model(X)
      # Calculate Probabilities (logits -> prediction probabilities)
      y_prob = y_logits.softmax(dim=1).squeeze()
      y_pred = y_prob.argmax(dim=0)
      # Append as CPU
      pred_probs.append(y_prob.cpu())
      pred_label.append(y_pred.item())
      data_path.append(path)

  true_label = data.targets

  return data, pred_probs, pred_label, true_label, data_path


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: list[str],
                        image_size: tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
  # Open Image
  img = Image.open(image_path)

  # Create transformation for image (if one doesn't exist)
  if transform is not None:
    image_transform = transform
  else:
    image_transform = transforms.Compose([transforms.Resize(size=image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  ### Predict on image ###
  # Get model to device
  model.to(device)

  # Turn on model evaluation model and inference mode
  model.eval()
  with torch.inference_mode():
    # Transform and add an extra dimenion to image (moreddel requires samples in [batch, color_channels, height, width])
    transformed_image = image_transform(img).unsqueeze(dim=0)
    # Make a predciton on image with an extra dimension and send it to the target device
    pred_logits = model(transformed_image.to(device))
    pred_probs = pred_logits.softmax(dim=1)
    pred_label = pred_probs.argmax(dim=1)

  # Plot out fiture with results in Title
  plt.figure()
  plt.imshow(img)
  plt.title(f'Pred: {class_names[pred_label]} | Prob: {pred_probs.max():.3f}')
  plt.axis(False)
