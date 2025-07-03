import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(PerceptualLoss, self).__init__()
        # Cargar VGG16 pre-entrenada, solo la parte convolucional
        vgg = models.vgg16(weights='DEFAULT').features.to(device).eval()

        # Congelar los parámetros de VGG
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Definir las capas de las que extraeremos características
        # Estos son los índices de las capas ReLU en vgg.features
        self.feature_layers_indices = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        self.vgg_features = vgg
        self.loss_fn = nn.L1Loss()

    def forward(self, generated, real):
        # Es importante que las imágenes estén normalizadas como espera VGG
        # Esto puede requerir una normalización específica si no usas la estándar de ImageNet
        # Por ahora, asumimos que la normalización [0,1] es suficientemente buena para empezar.
        
        gen_features = {}
        real_features = {}
        
        x_gen, x_real = generated, real
        
        total_loss = 0.0
        
        # Iterar a través de las capas de VGG
        for name, layer in self.vgg_features._modules.items():
            x_gen = layer(x_gen)
            x_real = layer(x_real)
            
            if name in self.feature_layers_indices:
                total_loss += self.loss_fn(x_gen, x_real)

        return total_loss