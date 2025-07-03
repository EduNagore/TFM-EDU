#import pandas as pd
import torch
import numpy as np
import pickle

from utils.datos import torch_DataLoader_SR
from utils.Arquitectura_NN import SuperResAutoencoder, LatentTransformerSR, SimpleDecoder, MultiHeadLatentTransformer, UNetDecoder
from datasets import load_dataset
from utils.plot import plot_super_resolution_results
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
_, _, test_loader_lr = torch_DataLoader_SR(
    images_np,
    batch_size=config.batch_size,
    device='cpu', drop_last=False, shuffle=False, size=config.input_size) # shuffle=False para evaluación

# Tomamos solo el primer lote para un ejemplo rápido
# NOTA: Para una evaluación real, deberías iterar sobre todo el test_loader_lr
print("Tomando un lote del conjunto de test para la evaluación...")
X_batch, Y_batch = next(iter(test_loader_lr))

# --- 4. Crear y cargar el modelo ---
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
    decoder=simple_decoder,
    target_size=config.target_size
).to(device)

model_name = config.model_name_transformer
model_path = f'models/{model_name}/{model_name}_epoch_{config.epochs}.pth'
print(f"Cargando modelo desde: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))

print("\n--- Arquitectura del Modelo ---")
print(model)
print("-----------------------------\n")

# --- 5. Predicción ---
model.eval()
with torch.no_grad():
    # CORRECCIÓN 3: Mover el tensor existente al dispositivo, no crear uno nuevo
    inputs = X_batch.to(device)
    outputs = model(inputs)
    

plot_super_resolution_results(
    inputs_lr=X_batch,
    targets_hr=Y_batch,
    outputs_hr=outputs,
    model_name=model_name,
    num_images_to_save=5, 
    save_dir='results'
)

# --- 6. Cálculo de métricas y guardado de resultados ---
predictions = outputs.cpu().numpy()
targets_np = Y_batch.cpu().numpy()
# Ahora ambos son arrays de NumPy
rmse = ((predictions - targets_np)**2).mean() ** 0.5
print(f'RMSE en el lote de test: {rmse:.4f}')

# Cargar la información del entrenamiento
info_path = f'models/{model_name}/{model_name}_epoch_{config.epochs}_info.pkl'
with open(info_path, 'rb') as f:
    data = pickle.load(f)

results = {
    'model_name'    : model_name,
    'num_tr_layers' : config.num_tr_layers,
    'nhead'         : config.nhead,
    'd_model'       : config.d_model_transformer,
    'fun_act'       : config.fun_act,
    'out_act'       : config.out_act,
    'lr'            : config.lr,
    'epochs'        : config.epochs,
    'batch_size'    : config.batch_size,
    'checkpoints'   : config.checkpoints,
    'rmse_on_batch' : rmse,
    'trainning_date': data['trainning_date'],
    'trainning_time': data['time'],
}

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# CORRECCIÓN 6: Usar la variable model_name para el nombre del fichero
results_file = f'{results_dir}/{model_name}.pkl'
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f'Resultados de la evaluación guardados en: {results_file}')