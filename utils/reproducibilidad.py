# utils/reproducibilidad.py

import torch
import numpy as np
import random
import os

def set_seeds(seed_value=42):
    """
    Fija las semillas de los generadores de números aleatorios para asegurar la reproducibilidad.
    """
    print(f"--- Fijando semillas para reproducibilidad con seed = {seed_value} ---")
    
    # 1. Semilla para Python nativo
    random.seed(seed_value)
    
    # 2. Semilla para NumPy
    np.random.seed(seed_value)
    
    # 3. Semilla para PyTorch en CPU y GPU
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # Para múltiples GPUs

    # 4. Configuraciones para determinismo en CUDA (puede afectar al rendimiento)
    # Estas configuraciones fuerzan a cuDNN a usar algoritmos deterministas.
    # Puede hacer que el código sea más lento, pero es necesario para una reproducibilidad total en la GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. Para reproducibilidad en algunas operaciones de PyTorch (a partir de PyTorch 1.9)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Opcional, para ciertas operaciones