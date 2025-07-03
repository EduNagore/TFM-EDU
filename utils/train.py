import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import pickle 
import shutil

from tqdm import tqdm
from utils.plot import plot_NN_hist
from utils.loss_functions import PerceptualLoss

def fit(model, 
        num_epochs,
        train_dl,
        val_dl=None,
        loss_fun='L1', # Default a L1
        optim_name='Adam',
        lr=1e-3,
        gamma=None,
        l2_norm_w=None,
        use_perceptual_loss=False,
        perceptual_loss_weight=0.01,
        display=True,
        checkpoints=None,
        model_name=None):
    
    device = next(model.parameters()).device

    # --- Funciones de pérdida ---
    if loss_fun == 'MSE':
        reconstruction_loss_fn = nn.MSELoss()
    elif loss_fun == 'L1':
        reconstruction_loss_fn = nn.L1Loss()
    else:
        raise ValueError(f'loss function {loss_fun} does not exist.')
    
    if use_perceptual_loss:
        perceptual_loss_fn = PerceptualLoss(device=device)

    # --- Optimizador ---
    if optim_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_norm_w if l2_norm_w else 0)
    else:
        raise ValueError(f'optimizer {optim_name} does not exist.')
    
    if gamma is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
    dir = f'models/{model_name}'
    if os.path.exists(dir): shutil.rmtree(dir)
    os.makedirs(dir)
    
    train_loss = []
    val_loss = []
    t_init = time.time()

    for epoch in range(num_epochs):
        
        # --- Entrenamiento ---
        model.train()
        act_train_loss = 0.0
        total_data = 0
        
        for inputs, targets in tqdm(train_dl, disable=not display, desc=f'Training epoch {epoch+1}/{num_epochs}'):
            x, y = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(x, target_size=y.shape[-2:])
            
            reconstruction_loss = reconstruction_loss_fn(outputs, y)
        
            if use_perceptual_loss:
                p_loss = perceptual_loss_fn(outputs, y)
                total_loss = reconstruction_loss + perceptual_loss_weight * p_loss
            else:
                total_loss = reconstruction_loss

            total_loss.backward()
            optimizer.step()

            # CORRECCIÓN 1: Registrar la pérdida de reconstrucción para consistencia
            act_train_loss += reconstruction_loss.item() * x.shape[0]
            total_data += x.shape[0]
        
        # CORRECCIÓN 2: Calcular métrica final de entrenamiento
        epoch_train_loss = act_train_loss / total_data
        if loss_fun == 'MSE':
            epoch_train_loss = epoch_train_loss**0.5
        train_loss.append(epoch_train_loss)
        
        # --- Validación ---
        if val_dl is not None:
            model.eval()
            act_val_loss = 0.0
            total_data = 0
            
            with torch.no_grad():
                for inputs, targets in tqdm(val_dl, disable=not display, desc=f'Testing  epoch {epoch+1}/{num_epochs}'):
                    x, y = inputs.to(device), targets.to(device)
                    outputs = model(x, target_size=y.shape[-2:])
                    
                    # CORRECCIÓN 3: Usar la función de pérdida correcta en validación
                    loss = reconstruction_loss_fn(outputs, y)
                    act_val_loss += loss.item() * x.shape[0]
                    total_data += x.shape[0]
            
            # CORRECCIÓN 4: Calcular métrica final de validación
            epoch_val_loss = act_val_loss / total_data
            if loss_fun == 'MSE':
                epoch_val_loss = epoch_val_loss**0.5
            val_loss.append(epoch_val_loss)
        
        # ... (resto del código para guardar checkpoints, etc., sin cambios) ...
        if gamma is not None:
            scheduler.step()
            
        t_act = time.time() - t_init
        
        if model_name is not None:
            label = None
            if (epoch + 1) == num_epochs:
                label = ''
            elif checkpoints is not None and (epoch + 1) % checkpoints == 0:
                label = f'. In process: {int(100*(epoch+1)/num_epochs)} %'
                
            if label is not None:
                file_name = f'{model_name}_epoch_{epoch+1}'
                plot_NN_hist(train_loss, val_loss, save=f'{model_name}', info=label)
                torch.save(model.state_dict(), f'{dir}/{file_name}.pth')
                
                info = {
                    'trainning_date': t_init,
                    'time': t_act,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
                with open(f'{dir}/{file_name}_info.pkl', 'wb') as f:
                    pickle.dump(info, f)