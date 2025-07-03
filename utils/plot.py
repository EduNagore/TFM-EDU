import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F



def plot_NN_hist(train_loss, val_loss = None, dir = None, save = None, info = ''):
    
    plt.figure()
    
    plt.semilogy(range(1, len(train_loss)+1), train_loss, label = 'Train loss')
    if val_loss is not None:
        plt.semilogy(range(1, len(train_loss)+1), val_loss, label = 'Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss during training{info}')
    plt.legend()
    plt.grid(which='minor')
    
    if save is not None:
        if dir is not None: dir = f'figures/loss/{dir}'
        else: dir = 'figures/loss'
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(f'{dir}/{save}.png')
    #plt.show()
    plt.close()

def plot_super_resolution_results(inputs_lr, targets_hr, outputs_hr, model_name, num_images_to_save=5, save_dir='results'):
    """
    Guarda una comparativa de imágenes de superresolución en una fila.

    Para cada imagen, crea un plot con 3 sub-imágenes:
    1. Entrada original de baja resolución (e.g., 7x7), visualizada para ver píxeles.
    2. Salida generada por el modelo en alta resolución.
    3. Salida real de alta resolución (target).
    """
    plot_subdir = os.path.join(save_dir, model_name)
    print(f"Generando {num_images_to_save} imágenes de comparación en '{plot_subdir}'...")

    os.makedirs(plot_subdir, exist_ok=True)

    inputs_lr = inputs_lr.cpu().detach()
    targets_hr = targets_hr.cpu().detach()
    outputs_hr = outputs_hr.cpu().detach()

    num_to_plot = min(num_images_to_save, inputs_lr.shape[0])

    for i in range(num_to_plot):
        # --- Preparar las 3 imágenes para el plot ---

        # 1. Imagen de entrada LR original (e.g., 7x7)
        img_lr_orig_np = inputs_lr[i].permute(1, 2, 0).numpy()

        # 2. Imagen generada (alta resolución)
        img_gen_np = outputs_hr[i].permute(1, 2, 0).numpy()
        
        # 3. Imagen objetivo (alta resolución)
        img_hr_np = targets_hr[i].permute(1, 2, 0).numpy()
        
        # Normalizar todas a [0, 1] para un display correcto
        img_lr_orig_np = np.clip(img_lr_orig_np, 0, 1)
        img_gen_np = np.clip(img_gen_np, 0, 1)
        img_hr_np = np.clip(img_hr_np, 0, 1)

        # --- Crear el plot (1 fila, 3 columnas) ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Resultados de Super-Resolución para {model_name} - Imagen {i}', fontsize=16)

        # 1. Entrada LR Original (7x7)
        axes[0].imshow(img_lr_orig_np, interpolation='nearest')
        axes[0].set_title(f'Entrada LR Original ({img_lr_orig_np.shape[0]}x{img_lr_orig_np.shape[1]})')
        axes[0].grid(True, which='both', color='w', linewidth=0.5, alpha=0.5)
        axes[0].tick_params(which='both', length=0)
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        
        # 2. Salida Generada (Modelo)
        axes[1].imshow(img_gen_np)
        axes[1].set_title('Salida Generada (Modelo)')
        axes[1].axis('off')

        # 3. Salida Esperada (Real)
        axes[2].imshow(img_hr_np)
        axes[2].set_title('Salida Esperada (Real)')
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajustar para que el suptitle no se solape
        
        # Guardar la figura
        save_path = os.path.join(plot_subdir, f"{model_name}_comparacion_{i}.png")
        plt.savefig(save_path)
        plt.close(fig) 

    print(f"Imágenes de comparación guardadas correctamente.")


def plot_autoencoder_results(inputs, outputs, model_name, num_images_to_save=5, save_dir='results'):
    """
    Guarda una comparativa de imágenes de un autoencoder.

    Para cada imagen, crea un plot con 2 sub-imágenes:
    1. Imagen de entrada original.
    2. Imagen reconstruida por el modelo.

    Args:
        inputs (torch.Tensor): Lote de imágenes de entrada (B, C, H, W).
        outputs (torch.Tensor): Lote de imágenes reconstruidas (B, C, H, W).
        model_name (str): Nombre del modelo, usado para nombrar los ficheros.
        num_images_to_save (int): Número de imágenes del lote a guardar.
        save_dir (str): Directorio donde se guardarán las imágenes.
    """
    plot_subdir = os.path.join(save_dir, model_name)
    print(f"Generando {num_images_to_save} imágenes de reconstrucción en '{plot_subdir}'...")

    os.makedirs(plot_subdir, exist_ok=True)
    
    inputs = inputs.cpu().detach()
    outputs = outputs.cpu().detach()

    num_to_plot = min(num_images_to_save, inputs.shape[0])

    for i in range(num_to_plot):
        # --- Preparar las 2 imágenes para el plot ---
        img_in_np = inputs[i].permute(1, 2, 0).numpy()
        img_out_np = outputs[i].permute(1, 2, 0).numpy()

        # Normalizar a [0, 1] para un display correcto
        img_in_np = np.clip(img_in_np, 0, 1)
        img_out_np = np.clip(img_out_np, 0, 1)

        # --- Crear el plot (1 fila, 2 columnas) ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Resultados para {model_name} - Imagen {i}', fontsize=16)

        axes[0].imshow(img_in_np)
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')

        axes[1].imshow(img_out_np)
        axes[1].set_title('Imagen Reconstruida')
        axes[1].axis('off')

        # Guardar la figura
        save_path = os.path.join(plot_subdir, f"reconstruccion_imagen_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(f"Imágenes de reconstrucción guardadas correctamente.")    