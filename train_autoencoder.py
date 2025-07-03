#import pandas as pd
import torch
import numpy as np

from utils.datos import torch_DataLoader
from utils.Arquitectura_NN import Autoencoder, SimpleDecoder, ResNet34Encoder, SimpleDecoder2, UNetAutoencoder
from utils.train import fit
from datasets import load_dataset
import os
from utils.reproducibilidad import set_seeds

set_seeds(42)  # Fijar semilla para reproducibilidad

# Easy import
#from config.config_1 import *

# Import as a more versatil way
import  sys
from    importlib               import import_module


# --- 1. Cargar configuración ---
if len(sys.argv) != 2:
    print("Uso: python train_nn.py <numero_de_config>")
    print("Usando config_1 por defecto.")
    config_num = 1
else:
    config_num = int(sys.argv[1])

try:
    config = import_module(f"config.config_{config_num}")
    print(f"--- Cargada configuración: config_{config_num}.py ---")
except ImportError:
    print(f"Error: No se pudo encontrar el archivo de configuración 'config/config_{config_num}.py'")
    sys.exit(1)


# --- 2. Preparar dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# --- 3. Load data
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ds = load_dataset("yasimed/lung_cancer")
images = ds['train']['image']
target_size = (224, 224)

images_resized = [
    np.array(image.convert("RGB").resize(target_size)) for image in images
]
# Formato PyTorch: (B, C, H, W)
images_np = np.stack(images_resized).transpose(0, 3, 1, 2) 

# ## CORRECCIÓN Y MEJOR PRÁCTICA ##
# Normalizar los datos al rango [0, 1] convirtiendo a float y dividiendo por 255
images_np = images_np.astype(np.float32) / 255.0
# Convertir a tensor de PyTorch

train_loader, val_loader, test_loader = torch_DataLoader(
    images_np,
    batch_size=config.batch_size)


# --- 4. Crear y cargar el modelo ---

model = UNetAutoencoder(
    out_channels=config.decoder_out_channels
).to(device)

print(f"Modelo creado y movido a {device}.")
print(f"Número de parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


fit(
    model=model,
    num_epochs=config.epochs,
    train_dl=train_loader,
    val_dl=val_loader, # Es mejor usar el set de validación durante el entrenamiento
    loss_fun='L1',
    optim_name=config.optimizer,
    lr=config.lr,
    display=True,
    checkpoints=config.checkpoints,
    model_name=config.model_name_autoencoder,
    use_perceptual_loss=True,
    perceptual_loss_weight=0.01
)

print("\n--- Entrenamiento finalizado ---")
