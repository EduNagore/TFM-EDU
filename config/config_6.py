# -- Nombre para guardar el modelo y los resultados --
model_name_transformer = "MutliHeadTransformer_v2"
model_name_autoencoder = "UNetAutoencoder_v2"
# -- Hiperparámetros de entrenamiento --
lr = 1e-4
epochs = 100
epochs_best_model = 400
epochs_autoencoder = 700
batch_size = 8
optimizer = 'Adam'
checkpoints = 20

# -- Configuración de la arquitectura --
# Parámetros para LatentTransformerSR
d_model_transformer = 256 # Dimensión del token (debe coincidir con el penúltimo canal)
num_tr_layers = 6         # Número de capas del Transformer
nhead = 8                 # Número de cabezas de atención (d_model % nhead debe ser 0)
n_coefficients = 64       # Coeficientes para los AttentionBlocks
latent_size = (7, 7)      # Tamaño del mapa de características de entrada (baja resolución)
num_tokens = latent_size[0] * latent_size[1] # 49
fun_act = 'relu'          # Función de activación interna
out_act = 'sigmoid'          # Función de activación de salida
input_size = (28, 28) # Tamaño de entrada del modelo (3 canales RGB, 224x224 píxeles)
target_size = (224, 224) # Tamaño de salida del modelo (3 canales RGB, 224x224 píxeles)
# Parámetros para SimpleDecoder
decoder_in_channels = 512 # Debe coincidir con la salida del transformer
decoder_out_channels = 3  # 3 canales para imagen RGB