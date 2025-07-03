# -- Nombre para guardar el modelo y los resultados --
model_name_transformer = "Transformer_v2"
model_name_autoencoder = "Autoencoder2_v1"
# -- Hiperparámetros de entrenamiento --
lr = 1e-4
epochs = 500
epochs_best_model = 250
epochs_autoencoder = 500
batch_size = 8
optimizer = 'Adam'
checkpoints = 10

# -- Configuración de la arquitectura --
# Parámetros para LatentTransformerSR
d_model_transformer = 256 # Dimensión del token (debe coincidir con el penúltimo canal)
num_tr_layers = 4         # Número de capas del Transformer
nhead = 8                 # Número de cabezas de atención (d_model % nhead debe ser 0)
n_coefficients = 64       # Coeficientes para los AttentionBlocks
latent_size = (7, 7)      # Tamaño del mapa de características de entrada (baja resolución)
num_tokens = latent_size[0] * latent_size[1] # 49
fun_act = 'relu'          # Función de activación interna
out_act = 'sigmoid'          # Función de activación de salida

# Parámetros para SimpleDecoder
decoder_in_channels = 512 # Debe coincidir con la salida del transformer
decoder_out_channels = 3  # 3 canales para imagen RGB