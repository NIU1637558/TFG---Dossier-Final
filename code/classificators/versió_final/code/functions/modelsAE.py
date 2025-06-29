import torch
import torch.nn as nn
import torch.optim as optim

class TextAutoEncoder(nn.Module):
    def __init__(self, input_dim=3*384, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

#### emb2emb AE #####
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, latent_dim=128, num_layers=1):
        super().__init__()
        # Encoder LSTM bidireccional
        self.lstm_encoder = nn.LSTM(input_size=input_dim,
                                   hidden_size=hidden_dim,
                                   num_layers=num_layers,
                                   bidirectional=True,
                                   batch_first=True)
        # Capa para obtener el embedding final (concatena último hidden state y cell state)
        self.to_latent = nn.Linear(2 * hidden_dim * num_layers * 2, latent_dim)  # *2 por bidireccional
        
        # Decoder (similar al tuyo pero adaptado)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Encoder
        _, (hidden, cell) = self.lstm_encoder(x)
        # Concatenar últimos hidden y cell states de todas las capas y direcciones
        hidden = hidden.permute(1, 0, 2).reshape(hidden.size(1), -1)
        cell = cell.permute(1, 0, 2).reshape(cell.size(1), -1)
        z = torch.cat([hidden, cell], dim=1)
        z = self.to_latent(z)
        
        # Decoder
        x_rec = self.decoder(z)
        return x_rec, z

class OffsetNetwork(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Ratio extraction network
        self.ratio_net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1),  # Aplica sobre (A,B) o (C,D)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * latent_dim, latent_dim)
        )
        # Conformity mapping network
        self.conformity_net = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
            
    def forward(self, A, B, C):
        # Calcular ratio A:B
        AB = torch.stack([A, B], dim=1)  # (batch, 2, latent_dim)
        ratio = self.ratio_net(AB)
        
        # Mapear ratio + C a D
        D = self.conformity_net(torch.cat([ratio, C], dim=1))
        return D



def add_noise(embeddings, noise_factor=0.1):
    noise = torch.randn_like(embeddings) * noise_factor
    return embeddings + noise

def train_autoencoder(all_text_embeddings, latent_dim=128, n_epochs=50, batch_size=64, lr=1e-3, noise_factor=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dividir embeddings en train y test
    X_train, X_test = train_test_split(all_text_embeddings, test_size=0.1, random_state=42)

    # Convertir a tensores
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    # DataLoader
    train_loader = DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test_tensor, batch_size=batch_size, shuffle=False)

    # Crear modelo
    input_dim = all_text_embeddings.shape[-1]  # Asumimos shape (n_samples, seq_len, input_dim)
    model = LSTMAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # W&B init
    initialize_wandb(args=None, name_model=f"LSTM_AE_latent{latent_dim}", nepoch=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Añadir ruido a los embeddings de entrada
            noisy_batch = add_noise(batch, noise_factor)
            
            output, _ = model(noisy_batch)
            loss = criterion(output, batch)  # Comparar con el original sin ruido
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        scheduler.step()

        # Evaluar en test set
        model.eval()
        with torch.no_grad():
            test_losses = []
            for batch in test_loader:
                batch = batch.to(device)
                output, _ = model(batch)
                loss = criterion(output, batch)
                test_losses.append(loss.item())
            avg_test_loss = np.mean(test_losses)

        # Log en wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss
        })

        print(f"[{epoch+1}/{n_epochs}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    return model, test_loader

# Función para entrenar el Offset Network
def train_offset_network(encoder, analogy_dataset, latent_dim=128, n_epochs=50, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preparar datos de analogías (A,B,C,D)
    # asumimos que analogy_dataset contiene tuplas (A,B,C,D)
    A_data = [encoder(torch.FloatTensor(analogy[0]).unsqueeze(0).to(device))[1] for analogy in analogy_dataset]
    B_data = [encoder(torch.FloatTensor(analogy[1]).unsqueeze(0).to(device))[1] for analogy in analogy_dataset]
    C_data = [encoder(torch.FloatTensor(analogy[2]).unsqueeze(0).to(device))[1] for analogy in analogy_dataset]
    D_data = [torch.FloatTensor(analogy[3]).unsqueeze(0).to(device) for analogy in analogy_dataset]
    
    # Convertir a tensores
    A_tensor = torch.cat(A_data, dim=0)
    B_tensor = torch.cat(B_data, dim=0)
    C_tensor = torch.cat(C_data, dim=0)
    D_tensor = torch.cat(D_data, dim=0)
    
    # DataLoader
    dataset = torch.utils.data.TensorDataset(A_tensor, B_tensor, C_tensor, D_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modelo
    offset_net = OffsetNetwork(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(offset_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        offset_net.train()
        losses = []
        
        for A, B, C, D in train_loader:
            optimizer.zero_grad()
            D_pred = offset_net(A, B, C)
            loss = criterion(D_pred, D)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        avg_loss = np.mean(losses)
        print(f"Offset Network Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
    return offset_net


### w2v + rnn ###
#### 3. RNN MODEL ####
class RNNEmbedder(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=32):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(emb_dim, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.rnn(x)
        return hidden[-1]

import torch
import torch.nn as nn

class Word2VecRNNEmbedder(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=64, bidirectional=True):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Fijamos Word2Vec

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (batch_size, seq_len)
        x_embed = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        _, (h_n, _) = self.rnn(x_embed)

        if self.bidirectional:
            # Concatenamos los últimos estados de ambas direcciones
            h_forward = h_n[-2, :, :]  # (batch, hidden_size)
            h_backward = h_n[-1, :, :]
            return torch.cat((h_forward, h_backward), dim=1)  # (batch, hidden_size * 2)
        else:
            return h_n[-1]  # (batch, hidden_size)
