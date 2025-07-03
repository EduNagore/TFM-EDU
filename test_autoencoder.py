#import pandas as pd
import torch
import numpy as np
import pickle

from utils.datos import torch_DataLoader
from utils.Arquitectura_NN import SimpleDecoder, ResNet34Encoder, Autoencoder, SimpleDecoder2, UNetAutoencoder
from datasets import load_dataset
from utils.plot import plot_autoencoder_results
import os
import sys
from importlib import import_module

# --- 1. Cargar configuración ---
if len(sys.argv) != 2:
    print("Uso: python <nombre_script.py> <numero_de_config>")
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

# Usamos el test_loader para la evaluación
_, _, test_loader = torch_DataLoader(
    images_np,
    batch_size=config.batch_size,
    device='cpu', drop_last=False, shuffle=False) # shuffle=False para evaluación

# Tomamos solo el primer lote para un ejemplo rápido
# NOTA: Para una evaluación real, deberías iterar sobre todo el test_loader_lr
print("Tomando un lote del conjunto de test para la evaluación...")
X_batch, Y_batch = next(iter(test_loader))

# --- 4. Crear y cargar el modelo ---

model = UNetAutoencoder(
    out_channels=config.decoder_out_channels)
model.to(device)
model_name = config.model_name_autoencoder
model_path = f'models/{model_name}/{model_name}_epoch_{config.epochs_autoencoder}.pth'
print(f"Cargando modelo desde: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))

print("\n--- Arquitectura del Modelo ---")
print(model)
print("-----------------------------\n")

# --- 5. Predicción ---
model.eval()
with torch.no_grad():
    inputs = X_batch.to(device)
    outputs = model(inputs)


results_dir = 'results'
plot_autoencoder_results(
    inputs=X_batch,
    outputs=outputs,
    model_name=model_name,
    num_images_to_save=5,
    save_dir=results_dir
)   

# --- 6. Cálculo de métricas y guardado de resultados ---
predictions = outputs.cpu().numpy()
targets_np = Y_batch.cpu().numpy()

# Ahora ambos son arrays de NumPy
rmse = ((predictions - targets_np)**2).mean() ** 0.5
print(f'RMSE en el lote de test: {rmse:.4f}')

# Cargar la información del entrenamiento
info_path = f'models/{model_name}/{model_name}_epoch_{config.epochs_autoencoder}_info.pkl'
with open(info_path, 'rb') as f:
    data = pickle.load(f)

results = {
    'model_name'    : model_name,
    'in_channels'   : config.decoder_in_channels,
    'out_channels'  : config.decoder_out_channels,
    'epochs'        : config.epochs_autoencoder,
    'batch_size'    : config.batch_size,
    'optimizer'     : config.optimizer,
    'learning_rate' : config.lr,
    'rmse'          : rmse,
    'info'          : data
}
    

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# CORRECCIÓN 6: Usar la variable model_name para el nombre del fichero
results_file = f'{results_dir}/{model_name}.pkl'
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f'Resultados de la evaluación guardados en: {results_file}')