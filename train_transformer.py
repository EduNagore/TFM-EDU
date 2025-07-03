#import pandas as pd
import torch
import numpy as np

from utils.datos import torch_DataLoader_SR
from utils.Arquitectura_NN import Autoencoder, SuperResAutoencoder, LatentTransformerSR, SimpleDecoder, MultiHeadLatentTransformer, UNetDecoder
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

images_np = images_np.astype(np.float32) / 255.0
# Convertir a tensor de PyTorch

train_loader_lr, val_loader_lr, test_loader_lr = torch_DataLoader_SR(
    images_np,
    batch_size=config.batch_size,
    device='cpu', drop_last=False, shuffle=True, size=config.input_size) # shuffle=True para entrenamiento



# 4.1. Crear la parte del Transformer
transformer_encoder = MultiHeadLatentTransformer(
    d_model=config.d_model_transformer,
    num_tr_layers=config.num_tr_layers,
    nhead=config.nhead)

# 4.2. Crear la parte del Decoder
simple_decoder = UNetDecoder(
    out_channels=config.decoder_out_channels,
)

# 4.3. Ensamblar el modelo final
model = SuperResAutoencoder(
    transformer=transformer_encoder,
    decoder=simple_decoder
).to(device)

print(f"Modelo creado y movido a {device}.")
print(f"Número de parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

path_pesos_decoder = 'models/UNetAutoencoder_v2/UNetAutoencoder_v2_epoch_700.pth'
if os.path.exists(path_pesos_decoder):
    print(f"Cargando pesos desde: {path_pesos_decoder}")
    
    # Cargar el diccionario de estado completo del fichero
    state_dict_completo = torch.load(path_pesos_decoder, map_location=device)
    
    # Crear un nuevo diccionario de estado solo para el decoder
    state_dict_decoder = {}
    prefijo = "decoder." # Este es el prefijo que queremos quitar
    
    for key, value in state_dict_completo.items():
        # Nos quedamos solo con las claves que empiezan con "decoder."
        if key.startswith(prefijo):
            # Creamos la nueva clave quitando el prefijo
            new_key = key[len(prefijo):]
            state_dict_decoder[new_key] = value

    for param in model.decoder.parameters():
        param.requires_grad = False 

    if not state_dict_decoder:
        print(f"ADVERTENCIA: No se encontraron claves con el prefijo '{prefijo}' en el fichero de pesos. ¿Estás seguro de que este fichero contiene los pesos del decoder?")
    else:
        print("Diccionario de estado del decoder extraído y adaptado. Cargando en el modelo...")
        # Cargar el diccionario de estado limpio en el decoder
        # El argumento strict=True (por defecto) nos ayuda a asegurar que todas las claves coinciden.
        model.decoder.load_state_dict(state_dict_decoder)

else:
    print(f"ADVERTENCIA: No se encontró el fichero de pesos del decoder en {path_pesos_decoder}. Se entrenará desde cero.")


fit(
    model=model,
    num_epochs=config.epochs,
    train_dl=train_loader_lr,
    val_dl=val_loader_lr,
    loss_fun='L1',
    optim_name=config.optimizer,
    lr=config.lr,
    display=True,
    checkpoints=config.checkpoints,
    model_name=config.model_name_transformer
)

print("\n--- Entrenamiento finalizado ---")
