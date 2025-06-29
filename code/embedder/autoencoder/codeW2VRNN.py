#### IMPORTS ####
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import wandb
import pandas as pd
import csv

# embeddings
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Autoencoder
from modelsAE import TextAutoEncoder, LSTMAutoEncoder

# manual test
from sklearn.metrics.pairwise import cosine_similarity

from modelsAE import RNNEmbedder

#### WANDB CONFIGURATION ####
def initialize_wandb(args, name_model,nepoch):
    wandb.init(
        project="TFG",
        entity="andreu-mir-uab",
        name=name_model,
        config={
            "framework": "PyTorch",
            "model": name_model,
            "LOSS": "MSE",
            "activation": "ReLU",
            "optimizer": "Adam",
            "max_epochs": nepoch,
            "dataset": "CH_Total2"
        }
    )

#### LOAD DATASET ####
def load_and_balance_data():
    print("Loading and preprocessing original data...")
    original_data = pd.read_csv('/fhome/amir/TFG/data/CH_Total2.csv')

    # text columns
    text_cols = ['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'REASONFORCHANGE_trad']
    categorical_cols = [
        'STATUS', 'ITNCHGCOMCANAL', 'ITNESCENARIO',
        'ACTSTART', 'PMCHGTYPE', 'OWNERGROUP', 'PMCHGCONCERN',
        'ITDCLOSURECODE', 'PROBLEMCODE', 'ITDCHCREATEDBYGROUP',
        'ACTFINISH', 'FIRSTAPPRSTATUS', 'PMCHGAPPROVALSTATE',
        'ITNAMBITO'
    ]

    # drop non text columns
    original_data = original_data.drop(columns=categorical_cols)
    print("Columns remaining in original data:", original_data.columns)

    # Extract 20 samples for manual testing
    mtest_data = original_data.sample(n=20, random_state=42)

    # Drop mtest_data from original_data
    original_data = original_data.drop(mtest_data.index)

    
    return original_data, mtest_data


##### GET EMBEDDINGS ####
def get_W2V_embeddings_sequential(data, text_cols, vector_size=32, max_len=50):
    nltk.download('punkt')

    # Entrenamiento del modelo Word2Vec
    tokenized_texts = []
    for col in text_cols:
        texts = data[col].fillna("").astype(str).tolist()
        tokenized_texts.extend([word_tokenize(t.lower()) for t in texts])
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=5, min_count=2, workers=4)

    # Obtener embeddings secuenciales
    sequences = []
    for idx, row in data[text_cols].fillna("").astype(str).iterrows():
        combined = " ".join(row)
        tokens = word_tokenize(combined.lower())
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if len(vecs) < max_len:
            vecs.extend([np.zeros(vector_size)] * (max_len - len(vecs)))
        else:
            vecs = vecs[:max_len]
        sequences.append(vecs)

    return np.array(sequences), w2v_model

###############
#### dataset for rnn ####
class CHSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # List or array of shape [n_samples, seq_len, emb_dim]
        self.labels = labels        # List or array of shape [n_samples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.sequences[idx])  # [seq_len, emb_dim]
        y = torch.LongTensor([self.labels[idx]])    # [1]
        return x, y.squeeze()


######### TRAINING FUNCTION #########
def train_rnn_model(sequences, labels, hidden_dim=64, n_epochs=20, batch_size=32, lr=1e-3, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # División
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.1, random_state=42)

    train_dataset = CHSequenceDataset(X_train, y_train)
    test_dataset = CHSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Modelo
    embedding_dim = sequences[0].shape[1]
    model = RNNEmbedder(embedding_dim, hidden_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Wandb init
    initialize_wandb(args=None, name_model="RNN_TextClassifier", nepoch=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        avg_train_loss = train_loss / total
        acc = correct / total

        # Evaluación
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)

        avg_test_loss = test_loss / test_total
        test_acc = test_correct / test_total

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": acc,
            "test_loss": avg_test_loss,
            "test_acc": test_acc
        })

        print(f"Epoch [{epoch+1}/{n_epochs}] | Train Loss: {avg_train_loss:.4f} Acc: {acc:.4f} | Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.4f}")

    return model, test_loader

#### save model ####
def save_rnn_model(model, model_name="RNNEmbedder"):
    model_path = f'/fhome/amir/TFG/code/embedder/results/models/{model_name}.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
    print(f"✅ Modelo guardado en: {model_path}")
    return model_path

##### EVALUATE MODEL #####
def evaluate_model(test_loader, model):
    model.eval()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    total_mse = 0.0
    total_mae = 0.0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(next(model.parameters()).device)
            output = model(batch)
            total_mse += criterion_mse(output, batch).item() * batch.size(0)
            total_mae += criterion_mae(output, batch).item() * batch.size(0)
            count += batch.size(0)

    avg_mse = total_mse / count
    avg_mae = total_mae / count

    return avg_mse, avg_mae

##### SAVE METRICS #####
def save_ae_metrics(mse, mae, name_model):
    rmse = np.sqrt(mse)

    summary_metrics = {
        'Reconstruction MSE': mse,
        'Reconstruction MAE': mae,
        'Reconstruction RMSE': rmse
    }

    # Log en W&B
    wandb.log(summary_metrics)

    # Guardar en CSV
    csv_path = f'/fhome/amir/TFG/code/autoencoders/results/{name_model}_ae_metrics.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    headers = ['Metric', 'Value']
    rows = [[metric, f"{value:.6f}"] for metric, value in summary_metrics.items()]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    # Log W&B Table
    df_metrics = pd.DataFrame(rows, columns=headers)
    wandb.log({f"{name_model}_ae_metrics_table": wandb.Table(dataframe=df_metrics)})

    return csv_path

##### MAIN FUNCTION #####
def main():
    # 1. W&B Initialization
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'RNN_W2V_SEQ'
    nepoch = 150
    initialize_wandb(args=None, name_model=name, nepoch=nepoch)
    model_path = f'/fhome/amir/TFG/code/embedder/results/models/{name}.pth'
    csv_path = f'/fhome/amir/TFG/code/embedder/results/{name}_metrics.csv'

    # 2. Load data
    sys.stderr.write(" -------------------- 2. Load data ---------------------------\n")
    original_data, _ = load_and_balance_data()

    # 3. Get sequential W2V embeddings
    sys.stderr.write(" -------------------- 3. Get sequential W2V embeddings ---------------------------\n")
    text_cols = ['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'REASONFORCHANGE_trad']
    max_len = 50
    vector_size = 32
    sequences, w2v_model = get_W2V_embeddings_sequential(original_data, text_cols, vector_size=vector_size, max_len=max_len)

    # 4. Train RNN
    sys.stderr.write(" -------------------- 4. Train RNN ---------------------------\n")
    y_dummy = np.random.randint(0, 2, size=(sequences.shape[0],))  # Etiquetas dummy

    model, test_loader = train_rnn_model(
        sequences=sequences,
        labels=y_dummy,
        hidden_dim=64,
        n_epochs=nepoch,
        batch_size=64,
        lr=1e-3,
        num_classes=2
    )

    # 5. Save model
    sys.stderr.write(" -------------------- 5. Save model ---------------------------\n")
    save_rnn_model(model, model_name=name)

    print("All done! 🎉")
    sys.stderr.write('---------------------------------------------------------------------\n')
    # Finaliza W&B
    wandb.finish()
    sys.stderr.write(" -------------------- 6. Finalize WandB ---------------------------\n")
    sys.stderr.write(f"Model saved to {model_path}\n")
    sys.stderr.write(f"Metrics saved to {csv_path}\n") 
    sys.stderr.write("All done!\n")
    sys.stderr.write('---------------------------------------------------------------------\n')

if __name__ == "__main__":
    main()