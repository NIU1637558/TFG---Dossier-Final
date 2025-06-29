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

#### emb2emb lstm ae ###
import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, latent_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: BiLSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Capa de proyección a espacio latente más profunda
        self.encoder_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim * num_layers, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )

        # Decoder más profundo
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder BiLSTM
        _, (h_n, c_n) = self.encoder_lstm(x)
        h_cat = torch.cat([h_n, c_n], dim=2).view(batch_size, -1)
        z = self.encoder_fc(h_cat)

        # Añadir ruido gaussiano
        noise = torch.randn_like(z) * 0.1
        z_noisy = z + noise

        # Decoder MLP
        x_rec = self.decoder(z_noisy)
        return x_rec


## RNN for embedding ##
#### W2V + RNN ###
import torch
import torch.nn as nn

class RNNEmbedder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, rnn_type='lstm'):
        super(RNNEmbedder, self).__init__()
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type debe ser 'lstm' o 'gru'")

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        if self.rnn_type == 'lstm':
            out, (hn, _) = self.rnn(x)
        else:
            out, hn = self.rnn(x)
        
        # Usamos el último hidden state como representación de la frase
        final_hidden = hn[-1]
        output = self.fc(final_hidden)
        return output