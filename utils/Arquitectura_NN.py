import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResNet34Encoder(nn.Module):
    """
    Encoder basado en ResNet-34 preentrenado.
    Retornará el último feature map convolucional,
    es decir [B, 512, H/32, W/32] si la entrada es [Batch, 3(RGB), Alto, Ancho].
    """
    def __init__(self):
        super().__init__()
        base_model = models.resnet34(weights='DEFAULT')
        # Cortamos la red antes del pooling y capa fully-connected:
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        self.out_channels = 512  # ResNet-34 produce 512 canales en el bloque final
    
    def forward(self, x):
        """
        Retorna el mapa de características final.
        Para entrada 224x224 => salida 7x7 (espacial) con 512 canales
        """
        return self.encoder(x)  # [B, 512, H/32, W/32]




class SimpleDecoder(nn.Module):
    """
    Decoder sencillo que hace upsample progresivo
    x2 en 5 pasos (factor total x32).
    """
    def __init__(self, in_channels=512, out_channels=3):
        super(SimpleDecoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )
    

    def forward(self, x):
        x = self.layer1(x)   # [B, 256, H/16,  W/16 ]
        x = self.layer2(x)   # [B, 128, H/8,   W/8  ]
        x = self.layer3(x)   # [B, 64,  H/4,   W/4  ]
        x = self.layer4(x)   # [B, 32,  H/2,   W/2  ]
        x = self.layer5(x)   # [B, 3,   H,     W    ]
        return x

class SimpleDecoder2(nn.Module):
    def __init__(self, in_channels=512, out_channels=3):
        super(SimpleDecoder2, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class AttentionBlock(nn.Module):
    """
    Attention block con parámetros aprendibles.

    F_g: número de canales en la señal “gate” (lo generado en esta etapa).
    F_l: número de canales en la skip connection (entrada anterior).
    n_coefficients: número de mapas intermedios para calcular la atención.
    """
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        gate: tensor generado en esta etapa, tamaño [B, F_g, H, W]
        skip_connection: tensor de la entrada anterior, tamaño [B, F_l, H, W]
        Devuelve skip_connection * mapa_de_atención, es decir [B, F_l, H, W].
        """
        g1 = self.W_gate(gate)                   # → [B, n_coefficients, H, W]
        x1 = self.W_x(skip_connection)           # → [B, n_coefficients, H, W]
        psi = self.relu(g1 + x1)                 # → [B, n_coefficients, H, W]
        psi = self.psi(psi)                      # → [B, 1, H, W], valores en (0,1)
        out = skip_connection * psi              # → [B, F_l, H, W]
        return out


class LatentTransformerSR(nn.Module):
    """
    Transformer híbrido para superresolución:
      - Escala progresivamente canales: 3→32→64→128→256
      - Insertamos un bloque Transformer sobre el latente 7x7 (256 canales)
      - Proyectamos finalmente a 512 canales y hacemos una última atención residual.
    La salida es [B, 512, 7, 7].
    """
    def __init__(self, n_coefficients=64,
                 d_model_transformer=256, nhead=8, num_tr_layers=2, num_tokens=49, fun_act='relu', out_act='relu'):
        super(LatentTransformerSR, self).__init__()

        # Definimos la “cadena” de: [3→32, 32→64, 64→128, 128→256]
        self.channels = [3, 32, 64, 128, 256, 512]
        num_etapas_enc = len(self.channels) - 2  # 4 etapas

        self.convs = nn.ModuleList()
        self.att_blocks = nn.ModuleList()
        self.att_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        if fun_act is None or fun_act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif fun_act == 'tanh':
            self.act = nn.Tanh()
        elif fun_act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif fun_act == 'softmax':
            self.act = nn.Softmax()
        elif fun_act == 'softsign':
            self.act = nn.Softsign()
        elif fun_act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError(f"Función de activación '{fun_act}' no existe.")
        
        if out_act is None or out_act == 'relu':
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == 'tanh':
            self.out_act = nn.Tanh()
        elif out_act == 'sigmoid':
            self.out_act = nn.Sigmoid()
        elif out_act == 'softmax':
            self.out_act = nn.Softmax()
        elif out_act == 'softsign':
            self.out_act = nn.Softsign()
        elif out_act == 'leaky_relu':
            self.out_act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError(f"Función de activación '{out_act}' no existe.")

        # Construcción de las primeras 4 etapas (3→32, 32→64, 64→128, 128→256):
        for i in range(num_etapas_enc):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            # Conv 1×1 que proyecta in_ch → out_ch
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True))
            # AttentionBlock: gate va a tener out_ch, skip tiene in_ch
            self.att_blocks.append(
                AttentionBlock(F_g=out_ch, F_l=in_ch, n_coefficients=n_coefficients)
            )
            # Conv 1×1 para proyectar el resultado de la atención (in_ch → out_ch)
            self.att_convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True))
            # BatchNorm sobre out_ch tras sumar conv + atención
            self.bns.append(nn.BatchNorm2d(out_ch))

        # Bloque Transformer que procesará el latente 7×7 con d_model = 256:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_transformer,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_tr_layers
        )
        # (Opcional): embedding posicional para x tokens de dimensión 256
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, d_model_transformer))

        # Después del Transformer, proyectamos a 512 canales:
        self.conv_final = nn.Conv2d(256, 512, kernel_size=1, bias=True)
        # Atención final para fusionar “skip (256 canales)” con “gate (512 canales)”
        self.att_block_final = AttentionBlock(F_g=512, F_l=256, n_coefficients=n_coefficients)
        self.att_conv_final = nn.Conv2d(256, 512, kernel_size=1, bias=True)
        self.bn_final = nn.BatchNorm2d(512)

        

    def forward(self, x):
        """
        Input:
          x: [B, 3, 7, 7]
        Output:
          out: [B, 512, 7, 7]
        """
        # --- Etapas 0–3: 3→32, 32→64, 64→128, 128→256 usando conv 1×1 + AttentionBlock + BN + ReLU
        x_prev = x
        for i in range(len(self.convs)):
            # 1) Conv 1×1 in_ch → out_ch
            x_base = self.convs[i](x_prev)  # → [B, out_ch, 7, 7]

            # 2) Atención: gate = x_base (out_ch), skip = x_prev (in_ch)
            att = self.att_blocks[i](gate=x_base, skip_connection=x_prev)  # → [B, in_ch, 7, 7]
            # 3) Proyectamos la atención a out_ch
            att_proj = self.att_convs[i](att)  # → [B, out_ch, 7, 7]

            # 4) Fusionamos residual y normalizamos
            x_sum = x_base + att_proj            # → [B, out_ch, 7, 7]
            x_sum = self.bns[i](x_sum)
            x_sum = self.act(x_sum)  # Aplicamos la función de activación

            # 5) Para la siguiente etapa:
            x_prev = x_sum

        # Ahora x_prev: [B, 256, 7, 7]

        # --- Paso Transformer sobre el latente 7×7, 256 canales ---
        b, c, h, w = x_prev.shape  # c = 256, h = w = 7
        # 1) Aplanamos a tokens: [B, 256, 49] → [B, 49, 256]
        x_flat = x_prev.view(b, c, h * w).permute(0, 2, 1)

        # 2) Sumamos embedding posicional (aprendible):
        x_flat = x_flat + self.pos_emb  # → [B, 49, 256]

        # 3) Aplicamos el TransformerEncoder (num_tr_layers capas):
        x_tr = self.transformer_encoder(x_flat)  # → [B, 49, 256]

        # 4) Volvemos al mapa 2D: [B, 49, 256] → [B, 256, 7, 7]
        x_tr_2D = x_tr.permute(0, 2, 1).contiguous().view(b, c, h, w)

        # --- Proyección final a 512 canales + Atención residual ---
        # 5) Conv 1×1 (256 → 512)
        x_base_final = self.conv_final(x_tr_2D)  # → [B, 512, 7, 7]

        # 6) Atención final: gate = x_base_final (512 ch), skip = x_tr_2D (256 ch)
        att_f = self.att_block_final(gate=x_base_final, skip_connection=x_tr_2D)  # → [B, 256, 7, 7]
        att_proj_f = self.att_conv_final(att_f)                                   # → [B, 512, 7, 7]

        # 7) Fusionamos, BN y ReLU
        out = x_base_final + att_proj_f  # → [B, 512, 7, 7]
        out = self.bn_final(out)
        out = self.out_act(out)

        return out



class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = ResNet34Encoder()
        self.decoder = SimpleDecoder()
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SuperResAutoencoder(nn.Module):
    """
    Autoencoder para superresolución que consta de:
      - Un bloque Transformer híbrido (LatentTransformerSR) que toma [B, 3, 7, 7]
        y produce un latente [B, 512, 7, 7].
      - Un decoder (SimpleDecoder) que recibe [B, 512, 7, 7] y reconstruye la imagen
        en alta resolución (por ejemplo, [B, 3, H, W]).
    """
    def __init__(self, transformer: nn.Module, decoder: nn.Module, target_size: tuple[int,int] = None):
        super(SuperResAutoencoder, self).__init__()
        self.transformer = transformer
        self.decoder = decoder
        self.target_size = target_size  # Tamaño de salida deseado, si se especifica
    def forward(self, x_lr: torch.Tensor):
        # 1) Pasar por el transformer para obtener los skips
        skips = self.transformer(x_lr)   # [s4, s3, s2, s1]
        # 2) Invocar al decoder con target_size
        out   = self.decoder(skips, target_size=self.target_size)
        return out
    

class ResNet34Encoder2(nn.Module):
    """
    Encoder basado en ResNet-34 preentrenado.
    Modificado para devolver mapas de características intermedios para una U-Net.
    """
    def __init__(self):
        super().__init__()
        base_model = models.resnet34(weights='DEFAULT')
        
        # Guardamos las capas del ResNet por separado para poder acceder a ellas
        self.initial_block = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        ) # Salida: 64 canales, H/4, W/4
        
        self.layer1 = base_model.layer1 # Salida: 64 canales, H/4, W/4
        self.layer2 = base_model.layer2 # Salida: 128 canales, H/8, W/8
        self.layer3 = base_model.layer3 # Salida: 256 canales, H/16, W/16
        self.layer4 = base_model.layer4 # Salida: 512 canales, H/32, W/32 (cuello de botella)

    def forward(self, x):
        """
        Retorna una lista de mapas de características, desde las capas iniciales hasta el final.
        """
        s0 = self.initial_block(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3) # Este es el cuello de botella
        
        # Devolvemos los mapas en orden, para que el decoder los use
        return [s4, s3, s2, s1]

class DecoderBlock(nn.Module):
    """
    Bloque del decoder de una U-Net.
    Realiza un upsampling y luego una convolución.
    Concatena el resultado con el skip connection del encoder.
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        
        # Concatenamos la salida del upsampling con el skip connection
        x_cat = torch.cat([x, skip], dim=1) # dim=1 es el eje de los canales
        
        x_out = self.conv_relu(x_cat)
        return x_out


class UNetDecoder(nn.Module):
    """
    Decoder que sigue la arquitectura U-Net.
    Recibe la lista de mapas de características del encoder.
    """
    def __init__(self, out_channels=3):
        super(UNetDecoder, self).__init__()
        
        # El cuello de botella del ResNet34 (s4) tiene 512 canales.
        # La capa de upsampling lo reducirá a 256.
        # Los skips tienen 256 (s3), 128 (s2), 64 (s1) canales.
        
        self.center = nn.Identity()
        
        # Block 1: upsample de 512 -> 256. Concat con skip s3 (256). middle_channels = 256 + 256 = 512
        self.dec4 = DecoderBlock(in_channels=512, middle_channels=256 + 256, out_channels=256)
        
        # Block 2: upsample de 256 -> 128. Concat con skip s2 (128). middle_channels = 128 + 128 = 256
        self.dec3 = DecoderBlock(in_channels=256, middle_channels=128 + 128, out_channels=128)
        
        # Block 3: upsample de 128 -> 64. Concat con skip s1 (64). middle_channels = 64 + 64 = 128
        self.dec2 = DecoderBlock(in_channels=128, middle_channels=64 + 64, out_channels=64)
        
        
        # Upsampling final para llegar a la resolución H/2
        self.up_final = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # Upsampling final para llegar a la resolución original H
        # La entrada aquí es de 32 canales, no hay skip connection
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(), # Buena idea añadir una no-linealidad
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1), # Usar k=3,p=1 preserva tamaño
            nn.Sigmoid()
        )
        
    def forward(self, skips, target_size=None):
        s4, s3, s2, s1 = skips
        x = self.center(s4)
        x = self.dec4(x, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.up_final(x)
        x = self.final_conv(x)      # [B,3, H_dec, W_dec]
        if target_size is not None:
            x = F.interpolate(x, size=target_size,
                              mode='bilinear',
                              align_corners=False)
        return x

class UNetAutoencoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        # Importante: Ahora pasas el número de canales de salida al decoder
        self.encoder = ResNet34Encoder2()
        self.decoder = UNetDecoder(out_channels=out_channels)
        
    def forward(self, x):
        skips = self.encoder(x)
        reconstruction = self.decoder(skips)
        return reconstruction



class MultiHeadLatentTransformer(nn.Module):
    """
    Transformer híbrido con embeddings de posición aprendibles flexibles
    para entradas de cualquier resolución H×W×3. Produce mapas de características:
    s4, s3, s2, s1, en las resoluciones esperadas por un decoder U-Net (sin unificar).
    """
    def __init__(self,
                 in_channels: int = 3,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_tr_layers: int = 6,
                 max_input_size: tuple[int, int] = (56, 56)
                ):
        super().__init__()
        H_max, W_max = max_input_size
        self.d_model = d_model

        # 1) PRE-ENCODER CONVOLUCIONAL
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=False)
        )

        # 2) EMBEDDINGS DE POSICIÓN APRENDIBLES (fila/columna)
        self.row_emb = nn.Parameter(torch.randn(H_max, d_model // 2))
        self.col_emb = nn.Parameter(torch.randn(W_max, d_model // 2))

        # 3) CUERPO DEL TRANSFORMER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_tr_layers
        )

        # 4) CABEZAS DE SALIDA
        self.head_s4 = nn.Conv2d(d_model, 512, kernel_size=3, padding=1)
        self.head_s3 = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(d_model // 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )
        self.head_s2 = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(d_model // 2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.head_s1 = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(d_model // 4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

    def forward(self, x_lr: torch.Tensor):
        """
        x_lr: Tensor [B, 3, H, W]
        returns: [s4, s3, s2, s1] en resoluciones adecuadas:
          s4 -> [B,512, H,   W]
          s3 -> [B,256,2*H, 2*W]
          s2 -> [B,128,4*H, 4*W]
          s1 -> [B, 64,8*H, 8*W]
        """
        B, C, H, W = x_lr.shape
        feats = self.pre_encoder(x_lr)               # [B, d_model, H, W]
        _, D, Hf, Wf = feats.shape

        # Positional embeddings dinamicos
        re = self.row_emb[:Hf]                       # [Hf, d/2]
        ce = self.col_emb[:Wf]                       # [Wf, d/2]
        grid = re.unsqueeze(1) + ce.unsqueeze(0)     # [Hf, Wf, d/2]
        pos = torch.cat([grid, grid], dim=-1)        # [Hf, Wf, d_model]
        pos = pos.view(1, Hf * Wf, D).to(feats.device)

        flat = feats.flatten(2).permute(0, 2, 1)      # [B, Hf*Wf, d_model]
        flat = flat + pos                            # suma out-of-place

        tr_out = self.transformer_encoder(flat)      # [B, Hf*Wf, d_model]
        tr_map = tr_out.permute(0, 2, 1).view(B, D, Hf, Wf)

        # Salidas por cabeza sin interpolación
        s4 = self.head_s4(tr_map)   # [B,512,   Hf,   Wf]
        s3 = self.head_s3(tr_map)   # [B,256, 2*Hf, 2*Wf]
        s2 = self.head_s2(tr_map)   # [B,128, 4*Hf, 4*Wf]
        s1 = self.head_s1(tr_map)   # [B, 64, 8*Hf, 8*Wf]

        return [s4, s3, s2, s1]