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
from modelsAE import TextAutoEncoder

# manual test
from sklearn.metrics.pairwise import cosine_similarity


#### WANDB CONFIGURATION ####
def initialize_wandb(args, name_model,nepoch):
    wandb.init(
        project="TFG",
        entity="andreu-mir-uab",
        name=name_model,
        config={
            "framework": "PyTorch",
            "model": name_model,
            "embedding": "all-MiniLM-L6-v2",
            "LOSS": "MSE",
            "activation": "ReLU",
            "optimizer": "Adam",
            "max_epochs": nepoch,
            "dataset": "CH_Total2"
        }
    )

#### DATA LOADER ####
class CHDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Convertir sparse matrix a dense array y luego a tensor
        x = torch.FloatTensor(self.X[idx].toarray().squeeze())
        y = torch.FloatTensor([self.y[idx]])
        return x, y

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

##### 1. w2v embeddings #####
def get_avg_embedding(text, model):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_W2V_embeddings(balanced_data, text_cols, indiv = False):
    # Cargar el modelo de Word2Vec
    nltk.download('punkt_tab')

    # Tokenize text
    tokenized_texts = []

    for col in text_cols:
        if col in og_data.columns:
            texts = og_data[col].fillna("").astype(str).tolist()
            tokenized_texts.extend([word_tokenize(text.lower()) for text in texts])

    # Embedding
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=32,
        window=5,
        min_count=2,
        workers=4
    )

    # get average embedding
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = np.array([get_avg_embedding(text, w2v_model) for text in texts])
            text_embeddings.append(embeddings)

    if indiv:
        return text_embeddings, w2v_model

    else:
        # Concatenar embeddings o crear array vacío si no hay columnas válidas
        all_text_embeddings = np.hstack(text_embeddings) if text_embeddings else np.zeros((len(balanced_data), 0))

        return all_text_embeddings, w2v_model
###############

###### 2. BERT embeddings #####
def get_BERT_embeddings(balanced_data, text_cols, indiv = False):
    # Preparar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # BERT models
    # 1- paraphrase-multilingual-MiniLM-L12-v2
    # 2- all-MiniLM-L6-v2
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Generar embeddings por columna de texto
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                device=device,
                normalize_embeddings=True
            )
            text_embeddings.append(embeddings)

    # guardar embeddings
    np.save('/fhome/amir/TFG/code/embedder/results/embeddings/AE1_BERT_embeddings.npy', text_embeddings)

    # # cargar embeddings
    # text_embeddings = np.load('/fhome/amir/TFG/code/embedder/results/embeddings/AE1_BERT_embeddings.npy')


    if indiv:
        return text_embeddings, model
    else:
        # Concatenar embeddings o crear array vacío si no hay columnas válidas
        all_text_embeddings = np.hstack(text_embeddings) if text_embeddings else np.zeros((len(balanced_data), 0))

        return all_text_embeddings, model

######### TRAINING FUNCTION #########
def train_autoencoder(all_text_embeddings, latent_dim=10, n_epochs=50, batch_size=64, lr=1e-3):
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
    input_dim = all_text_embeddings.shape[1]
    model = TextAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # W&B init
    initialize_wandb(args=None, name_model=f"AE_latent{latent_dim}", nepoch=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
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
                output = model(batch)
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

    return model.encoder, model, test_loader  # devolvemos también el autoencoder completo por si querés usar el decoder luego

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

##### MANUAL TEST #####
def manual_test_reconstruction_direct(mtest_data, text_cols, w2v_model, ae_model, device):
    ae_model.eval()
    ae_model.to(device)

    print("\n🔍 Manual AutoEncoder Reconstruction Test (embedding vs reconstructed):\n")

    for idx, row in mtest_data.iterrows():
        # Unir columnas de texto
        original_text = " ".join([str(row[col]) for col in text_cols if col in row])

        # Obtener embedding original (Word2Vec)
        orig_embedding = get_avg_embedding(original_text, w2v_model)
        if np.all(orig_embedding == 0):
            print(f"❌ Texto vacío o sin palabras en vocabulario en el índice {idx}. Saltando...\n")
            continue

        # Pasar por el AE
        orig_tensor = torch.tensor(orig_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            rec_embedding = ae_model(orig_tensor).cpu().numpy().squeeze()

        # Calcular métricas
        mse = mean_squared_error(orig_embedding, rec_embedding)
        mae = mean_absolute_error(orig_embedding, rec_embedding)
        cos_sim = cosine_similarity([orig_embedding], [rec_embedding])[0][0]

        # Mostrar resultados
        print("-" * 70)
        print(f"📝 Texto original:\n{original_text}")
        print(f"📐 MSE:  {mse:.6f}")
        print(f"📐 MAE:  {mae:.6f}")
        print(f"📐 Cosine Similarity: {cos_sim:.6f}")

    print("-" * 70)


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
    ## ------------- 1. Initialize WandB --------------------##
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'AE1_BERT'
    nepoch = 50
    initialize_wandb(args=None, name_model=name, nepoch=nepoch)

    ## ------------- 2. Load data --------------------##
    sys.stderr.write(" -------------------- 2. Load data ---------------------------\n")
    original_data, mtest_data = load_and_balance_data()

    # -------------- 3. Get embeddings --------------------##
    sys.stderr.write(" -------------------- 3. Get embeddings ---------------------------\n")
    if 'indiv' in name:
        indiv = True

    if 'BERT' in name:
        text_embeddings, emb_model = get_BERT_embeddings(original_data, text_cols=['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'REASONFORCHANGE_trad'], indiv = indiv)
    else:
        text_embeddings, emb_model = get_W2V_embeddings(original_data, text_cols=['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'REASONFORCHANGE_trad'], indiv = indiv)
    
    # -------------- 4. Train autoencoder --------------------##
    sys.stderr.write(" -------------------- 4. Train autoencoder ---------------------------\n")
    latent_dim = 10
    n_epochs = nepoch
    batch_size = 64
    lr = 1e-3

    # si es individual, entrenat 1 per cada camp
    if indiv:
        model1, autoencoder1, test_loader = train_autoencoder(text_embeddings[0], latent_dim=latent_dim, n_epochs=n_epochs, batch_size=batch_size, lr=lr)
        model2, autoencoder2, test_loader = train_autoencoder(text_embeddings[1], latent_dim=latent_dim, n_epochs=n_epochs, batch_size=batch_size, lr=lr)
        model3, autoencoder3, test_loader = train_autoencoder(text_embeddings[2], latent_dim=latent_dim, n_epochs=n_epochs, batch_size=batch_size, lr=lr)
    # sino tots junts
    else:
        model, autoencoder, test_loader = train_autoencoder(text_embeddings, latent_dim=latent_dim, n_epochs=n_epochs, batch_size=batch_size, lr=lr)

    # -------------- 5. Evaluate model --------------------##
    sys.stderr.write(" -------------------- 5. Evaluate model ---------------------------\n")
    if indiv:
        mse1, mae1 = evaluate_model(test_loader, autoencoder1)
        mse2, mae2 = evaluate_model(test_loader, autoencoder2)
        mse3, mae3 = evaluate_model(test_loader, autoencoder3)
    else:
        mse, mae = evaluate_model(test_loader, autoencoder)
    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

    # -------------- 6. Manual test --------------------##
    sys.stderr.write(" -------------------- 6. Manual test ---------------------------\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Realizar la prueba manual
    # manual_test_reconstruction_direct(mtest_data, ['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'REASONFORCHANGE_trad'], emb_model, autoencoder, device)
    
    # -------------- 7. Save metrics --------------------##
    sys.stderr.write(" -------------------- 6. Save metrics ---------------------------\n")
    if indiv:
        save_ae_metrics(mse1,mae1,f'{name}_text1')
        save_ae_metrics(mse2,mae2,f'{name}_text2')
        save_ae_metrics(mse3,mae3,f'{name}_text3')
    else:
        save_ae_metrics(mse, mae, name)

    # -------------- 8. Save model --------------------##
    sys.stderr.write(" -------------------- 7. Save model ---------------------------\n")
    # save encoder only
    if indiv:
        model_path1 = f'/fhome/amir/TFG/code/embedder/results/models/{name}_text1_model.pth'
        model_path2 = f'/fhome/amir/TFG/code/embedder/results/models/{name}_text2_model.pth'
        model_path3 = f'/fhome/amir/TFG/code/embedder/results/models/{name}_text3_model.pth'
    else:
        model_path = f'/fhome/amir/TFG/code/embedder/results/models/{name}_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(autoencoder.state_dict(), model_path)
    wandb.save(model_path)
    wandb.save(os.path.join(os.path.dirname(__file__), 'codeAE1.py'))
    wandb.finish()
    print(f"Model saved to {model_path}")
    print("All done! 🎉")

if __name__ == "__main__":
    main()
