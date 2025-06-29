import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP basic
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class MLP2(nn.Module):
    def __init__(self, input_size, sigmoid=True, dropout=0.5):
        super(MLP2, self).__init__()
        
        # Capa 1
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch Normalization
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Capa 2
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Capa 3
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.9)
        
        # Capa de salida
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Aplicar Batch Normalization
        x = self.relu1(x)
        x = self.dropout1(x)  # Aplicar Dropout
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x


class MLPAttention1(nn.Module):
    def __init__(self, input_size, sigmoid=True):
        super(MLPAttention1, self).__init__()
        self.sigmoid_flag = sigmoid

        # Capa de atención: genera pesos para cada input feature
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()  # valores entre 0 y 1 para ponderar cada característica
        )

        # Capa 1
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        # Capa 2
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # Capa 3
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        # Salida
        self.fc4 = nn.Linear(64, 1)
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Aplicar atención
        attention_weights = self.attention(x)
        x = x * attention_weights  # atención aplicada

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        if self.sigmoid_flag:
            x = self.output_sigmoid(x)
        return x

############ ATTENTION multihead ###############

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttentionMLP(nn.Module):
    def __init__(self, num_categorical, embedding_dim, dropout=0.5, num_heads=4, hybrid=True):
        super(HybridAttentionMLP, self).__init__()
        self.hybrid = hybrid  # flag para indicar si es el modelo híbrido
        # Proyecciones iniciales
        self.embedding_proj = nn.Linear(embedding_dim, 96)
        self.categorical_proj = nn.Linear(num_categorical, 17)
        
        # Combinar proyecciones
        self.combine_proj = nn.Linear(113, 128)

        # Bloques MLP
        self.fc1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        # Mecanismo de atención multihead
        self.embed_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Atención sobre categóricas
        self.cat_attention = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 1), # ern hybruid pasar vector abas de 32-->1
            nn.Softmax(dim=1)
        )
        
        # Alpha learnable: peso entre embed_attn y cat_attn
        self.attention_alpha = nn.Parameter(torch.tensor(0.6))  # inicializado en 0.6

        # Capa de salida
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_embed, x_cat):
        # Proyecciones
        embed_proj = self.embedding_proj(x_embed)
        cat_proj = self.categorical_proj(x_cat)
        
        # Concatenar features
        x = torch.cat([embed_proj, cat_proj], dim=1)  # (batch_size, 113)
        
        # Combinar y pasar por MLP
        x = self.combine_proj(x)
        
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))  # (batch_size, 32)
        
        # Atención embeddings
        x_reshaped = x.unsqueeze(1)  # (batch_size, 1, 32)
        embed_attn, _ = self.embed_attention(x_reshaped, x_reshaped, x_reshaped)
        embed_attn = embed_attn.squeeze(1)  # (batch_size, 32)
        
        # Atención categóricas
        cat_attn_weights = self.cat_attention(x)
        cat_attn = x * cat_attn_weights
        
        # Fusión de atenciones usando alpha aprendible
        alpha = torch.clamp(self.attention_alpha, 0.0, 1.0)  # asegura que alpha esté entre [0,1]
        x = alpha * embed_attn + (1.0 - alpha) * cat_attn # (batch_size, 32) --> això es el que rep el hybrid al concatenar outputs
        
        if self.hybrid:
            # Si es el modelo híbrido, no aplicar sigmoid aquí
            return x

        # Output
        x = self.fc4(x)
        return self.sigmoid(x)


### BAGGING ENSAMBLER ###
# bagging.py (nuevo archivo)

import torch
import torch.nn as nn
import numpy as np
from model_arquitectres.models import MLP2, HybridAttentionMLP

class BaggingEnsemble(nn.Module):
    def __init__(self, n_estimators=10, input_size=None, txt_shape=96, att=False, device='cpu'):
        super(BaggingEnsemble, self).__init__()
        self.n_estimators = n_estimators
        self.input_size = input_size
        self.txt_shape = txt_shape
        self.att = att
        self.device = device
        self.estimators = nn.ModuleList()
        
        # Inicializar los estimadores
        for _ in range(n_estimators):
            if att:
                model = HybridAttentionMLP(input_size - txt_shape, txt_shape, dropout=0.8, hybrid = False).to(device)
            else:
                model = MLP2(input_size, dropout=0.8).to(device)
            
            # Inicializar pesos
            self._init_weights(model)
            self.estimators.append(model)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Predicciones de todos los modelos
        all_preds = []
        
        if self.att:
            input_txt, input_cat = x[:, :self.txt_shape], x[:, self.txt_shape:]
            for model in self.estimators:
                preds = model(input_txt, input_cat)
                all_preds.append(preds)
        else:
            for model in self.estimators:
                preds = model(x)
                all_preds.append(preds)
        
        # Combinar predicciones (promedio)
        stacked_preds = torch.stack(all_preds, dim=0)
        avg_preds = torch.mean(stacked_preds, dim=0)
        
        return avg_preds
    
    def predict(self, x):
        """Predicción binaria por votación mayoritaria"""
        self.eval()
        with torch.no_grad():
            if self.att:
                input_txt, input_cat = x[:, :self.txt_shape], x[:, self.txt_shape:]
                all_votes = []
                for model in self.estimators:
                    outputs = model(input_txt, input_cat)
                    votes = (outputs > 0.5).float()
                    all_votes.append(votes)
            else:
                all_votes = []
                for model in self.estimators:
                    outputs = model(x)
                    votes = (outputs > 0.5).float()
                    all_votes.append(votes)
            
            # Votación mayoritaria
            stacked_votes = torch.stack(all_votes, dim=0)
            majority_vote = (torch.mean(stacked_votes, dim=0) > 0.5).float()
            
            return majority_vote

### HYBRID MODEL ###

# NEURONA (no-trainable)
class NeuronaModel(nn.Module):
    def __init__(self, model):
        super(NeuronaModel, self).__init__()
        self.model = model
        self.model.eval()

    def forward(self, x_embed, x_cat):
        with torch.no_grad():
            return self.model(x_embed, x_cat)

# NEURONA (trainable)
class NeuronaModelT(nn.Module):
    def __init__(self, model):
        super(NeuronaModelT, self).__init__()
        self.model = model

    def forward(self, x_embed, x_cat):
        return self.model(x_embed, x_cat)


# HIBRID MODEL
import torch
import torch.nn as nn

class HybridNeuralStacking(nn.Module):
    def __init__(self, neuron_map: dict, macro_txtshape: dict):
        super(HybridNeuralStacking, self).__init__()

        # Envolver modelos preentrenados en estructuras de PyTorch
        self.neuron_map = nn.ModuleDict({
            key: nn.ModuleList(model_list)
            for key, model_list in neuron_map.items()
        })

        self.macro_txtshape = macro_txtshape
        self.total_neurons = sum(len(models) for models in neuron_map.values())

        # Inicializar pesos de combinación con Xavier uniforme
        self.weights = nn.Parameter(torch.empty(self.total_neurons))
        nn.init.xavier_uniform_(self.weights.unsqueeze(1))  # Xavier init

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_dict):
        outputs = []
        idx = 0

        for key, neuron_list in self.neuron_map.items():
            x = inputs_dict[key]
            txt_dim = self.macro_txtshape[key]
            x_embed = x[:, :txt_dim]
            x_cat = x[:, txt_dim:]

            for neuron in neuron_list:
                y = neuron(x_embed, x_cat).squeeze(-1)
                outputs.append(y)
                idx += 1

        outputs = torch.stack(outputs, dim=1)  # (batch_size, num_neuronas)
        normalized_weights = torch.softmax(self.weights, dim=0)
        weighted_sum = (outputs * normalized_weights).sum(dim=1)

        # return self.sigmoid(weighted_sum)
        return weighted_sum  # sin sigmoid para el modelo híbrido

# hybrid model trainable
class HybridNeuralStackingT(nn.Module):
    def __init__(self, neuron_map: dict, macro_txtshape: dict):
        super(HybridNeuralStackingT, self).__init__()
        self.neuron_map = nn.ModuleDict({
            key: nn.ModuleList(model_list)
            for key, model_list in neuron_map.items()
        })
        self.macro_txtshape = macro_txtshape

        self.total_neurons = sum(len(models) for models in neuron_map.values())
        self.weights = nn.Parameter(torch.ones(self.total_neurons))

        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_dict, return_all=False):
        outputs = []
        idx = 0

        for key, neuron_list in self.neuron_map.items():
            x = inputs_dict[key]
            txt_dim = self.macro_txtshape[key]
            x_embed = x[:, :txt_dim]
            x_cat = x[:, txt_dim:]

            for neuron in neuron_list:
                y = neuron(x_embed, x_cat).squeeze(-1)
                outputs.append(y)
                idx += 1

        stacked_outputs = torch.stack(outputs, dim=1)  # (B, (nmodels,1))
        normalized_weights = torch.softmax(self.weights, dim=0)
        weighted_sum = (stacked_outputs * normalized_weights).sum(dim=1)
        # final_output = self.sigmoid(weighted_sum)
        final_output = weighted_sum

        if return_all:
            return final_output, stacked_outputs
        else:
            return final_output

### 3 maneres diferents de combinar els vectors de sortida de les neurones N * (b,32)

### 1. mitjana + linear 32 → 1
class HybridNeuralStackingT_v1(nn.Module):
    def __init__(self, neuron_map: dict, macro_txtshape: dict):
        super(HybridNeuralStackingT_v1, self).__init__()
        self.neuron_map = nn.ModuleDict({
            key: nn.ModuleList(model_list)
            for key, model_list in neuron_map.items()
        })
        self.macro_txtshape = macro_txtshape
        self.output_layer = nn.Linear(32, 1)  # última capa tras la media

    def forward(self, inputs_dict, return_all=False):
        outputs = []

        for key, neuron_list in self.neuron_map.items():
            x = inputs_dict[key]
            txt_dim = self.macro_txtshape[key]
            x_embed, x_cat = x[:, :txt_dim], x[:, txt_dim:]

            for neuron in neuron_list:
                vec = neuron(x_embed, x_cat)  # (B, 32)
                outputs.append(vec)

        stacked = torch.stack(outputs, dim=0)  # (n_models, B, 32)
        avg = stacked.mean(dim=0)              # (B, 32)
        out = self.output_layer(avg)           # (B, 1)
        
        if return_all:
            individual_outputs = [self.output_layer(vec).squeeze(-1) for vec in stacked]  # list of (B,)
            individual_outputs = torch.stack(individual_outputs, dim=0)  # (n_models, B)
            return out.squeeze(-1), individual_outputs  # (n_models, B)
        else:
            return out.squeeze(-1)

### 2. (n,b,32) → (b,32) → (b,1) (fusion vertical)
class HybridNeuralStackingT_v2(nn.Module):
    def __init__(self, neuron_map: dict, macro_txtshape: dict):
        super(HybridNeuralStackingT_v2, self).__init__()
        self.neuron_map = nn.ModuleDict({
            key: nn.ModuleList(model_list)
            for key, model_list in neuron_map.items()
        })
        self.macro_txtshape = macro_txtshape

        self.total_neurons = sum(len(models) for models in neuron_map.values())  # 12 en tu ejemplo

        self.fusion_conv = nn.Conv1d(in_channels=self.total_neurons, out_channels=1, kernel_size=1)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, inputs_dict, return_all=False):
        outputs = []

        for key, neuron_list in self.neuron_map.items():
            x = inputs_dict[key]
            txt_dim = self.macro_txtshape[key]
            x_embed, x_cat = x[:, :txt_dim], x[:, txt_dim:]

            for neuron in neuron_list:
                vec = neuron(x_embed, x_cat)  # (B, 32)
                outputs.append(vec)

        stacked = torch.stack(outputs, dim=0)  # (n, B, 32)
        stacked = stacked.permute(1, 0, 2)  # (B, n, 32)
        # 1. convolució per fusionar vectors
        fused = self.fusion_conv(stacked)  # (B, 1, 32) # convolució de n->1
        fused = fused.squeeze(1)  # (B, 32)
        # 2. capa de out
        out = self.output_layer(fused)  # (B, 1)

        if return_all:
            individual_outputs = [self.output_layer(vec).squeeze(-1) for vec in stacked]  # list of (B,)
            individual_outputs = torch.stack(individual_outputs, dim=0)  # (n_models, B)
            return out.squeeze(-1), individual_outputs.permute(1,0)  # (B, n_models)
        else:
            return out.squeeze(-1)


### 3. (n,b,32) → (n,b,1) → (b,1) (fusion horizontal)
class HybridNeuralStackingT_v3(nn.Module):
    def __init__(self, neuron_map: dict, macro_txtshape: dict):
        super(HybridNeuralStackingT_v3, self).__init__()
        self.neuron_map = nn.ModuleDict({
            key: nn.ModuleList(model_list)
            for key, model_list in neuron_map.items()
        })
        self.macro_txtshape = macro_txtshape

        self.total_neurons = sum(len(models) for models in neuron_map.values())  # 12 en tu caso

        self.output_layer = nn.Linear(32, 1)
        self.fusion_conv = nn.Conv1d(in_channels=self.total_neurons, out_channels=1, kernel_size=1)

    def forward(self, inputs_dict, return_all=False):
        outputs = []

        for key, neuron_list in self.neuron_map.items():
            x = inputs_dict[key]
            txt_dim = self.macro_txtshape[key]
            x_embed, x_cat = x[:, :txt_dim], x[:, txt_dim:]

            for neuron in neuron_list:
                vec = neuron(x_embed, x_cat)  # (B=7, 32)
                outputs.append(vec)

        stacked = torch.stack(outputs, dim=0)  # (n, B, 32)

        # 1. Capa de out
        out_vectors = [self.output_layer(vec).squeeze(-1) for vec in outputs]  # list de (B,)
        out_stacked = torch.stack(out_vectors, dim=0)  # (n, B)
        out_stacked = out_stacked.permute(1, 0).unsqueeze(-1)  # (B, n, 1)
        # 2. convolució per fusionar vectors
        fused = self.fusion_conv(out_stacked)  # (B, 1, 1)
        fused = fused.squeeze(-1).squeeze(-1)   # (B,)

        if return_all:
            individual_outputs = [self.output_layer(vec).squeeze(-1) for vec in stacked]  # list of (B,)
            individual_outputs = torch.stack(individual_outputs, dim=0)  # (n_models, B)
            return fused, individual_outputs.permute(1,0)  # (B, n_models)
        else:
            return fused



### 4. Subred conv per fusionar vectors
class HybridNeuralStackingT_v4(nn.Module):
    def __init__(self, neuron_map: dict, macro_txtshape: dict):
        super(HybridNeuralStackingT_v4, self).__init__()
        self.neuron_map = nn.ModuleDict({
            key: nn.ModuleList(model_list)
            for key, model_list in neuron_map.items()
        })
        self.macro_txtshape = macro_txtshape

        self.total_neurons = sum(len(models) for models in neuron_map.values())
        
        # Red convolucional para fusionar vectores (N, 32) → 1
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.total_neurons, out_channels=64, kernel_size=1),  # (B, 64, 32)
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),  # (B, 32, 32)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 32, 1)
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1),  # (B, 1, 1)
        )

                # Red convolucional para fusionar vectores (N, 32) → 1
        self.conv_net2 = nn.Sequential(
            nn.Conv1d(in_channels=self.total_neurons, out_channels=16, kernel_size=1),  # (B, 64, 32)
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1),  # (B, 16, 16)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 8, 1)
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1),  # (B, 1, 1)
        )


        # Para aplicar individualmente sobre cada vec de tamaño (B, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, inputs_dict, return_all=False):
        outputs = []

        for key, neuron_list in self.neuron_map.items():
            x = inputs_dict[key]
            txt_dim = self.macro_txtshape[key]
            x_embed, x_cat = x[:, :txt_dim], x[:, txt_dim:]

            for neuron in neuron_list:
                vec = neuron(x_embed, x_cat)  # (B, 32)
                outputs.append(vec)

        stacked = torch.stack(outputs, dim=0)     # (N, B, 32)
        stacked = stacked.permute(1, 0, 2)        # (B, N, 32)

        # CNN para fusión total → (B, 1, 1) → (B,)
        x = self.conv_net(stacked)                # (B, 1, 1)
        out = x.squeeze(-1).squeeze(-1)           # (B,)

        if return_all:
            # Individual outputs usando output_layer sobre cada vec
            individual_outputs = [self.output_layer(vec).squeeze(-1) for vec in stacked.permute(1, 0, 2)]  # [(B,)]
            individual_outputs = torch.stack(individual_outputs, dim=1)  # (B, N)
            return out, individual_outputs.permute(1, 0)  # (N, B)
        else:
            return out
