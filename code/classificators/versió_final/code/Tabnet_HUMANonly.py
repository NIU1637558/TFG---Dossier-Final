import os
import sys
import pandas as pd
import numpy as np
import csv
import argparse
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                            f1_score, classification_report, confusion_matrix)
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import defaultdict
from imblearn.under_sampling import RandomUnderSampler
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import model
from pytorch_tabnet.tab_model import TabNetClassifier
from collections import Counter

# 1. Definición del Custom Dataset
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fails', type=bool, default=False, 
                       help='Guardar registros con predicciones fallidas')
    return parser.parse_args()

def initialize_wandb(args, name_model,nepoch, extra):
    wandb.init(
        project="TFG",
        entity="andreu-mir-uab",
        name=name_model,
        config={
            "framework": "PyTorch",
            "model": "TABNET1",
            "LOSS": "BCEweighted",
            "activation": "ReLU",
            "optimizer": "Adam",
            "max_epochs": nepoch,
            "patience": 10,
            "dataset": "CH_Total2",
            "balance_method": "RandomUnderSampling",
            "extras": extra,
            "save_fails": args.save_fails
        }
    )

def filter_out_RPA_rejected(dataframe):
    """
    Elimina filas donde REJECTED es RPA.
    """
    # filter ALL RPA
    mask = ~((dataframe['author'].str.upper() == 'RPA'))

    # # filter only REJECTED RPA
    # mask = ~((dataframe['REJECTED'] == True) & (dataframe['author'] == 'RPA'))

    return dataframe[mask].reset_index(drop=True)

def load_and_balance_data(extra):
    print("Loading and preprocessing original data...")
    original_data = pd.read_csv('/fhome/amir/TFG/data/CH_Total2.csv')
    target_col = 'REJECTED'

    # Filter out HUMAN rejected
    original_data = filter_out_RPA_rejected(original_data)

    # Sample n extra rows from original rejected False
    if extra > 0:
        rejected_false = original_data[original_data[target_col] == 0].sample(n=extra, random_state=42)
        original_data = original_data.drop(rejected_false.index)

    # Balance data (manteniendo índice original)
    undersampler = RandomUnderSampler()
    X_temp, y_temp = original_data.drop(columns=[target_col]), original_data[target_col]
    X_balanced, y_balanced = undersampler.fit_resample(X_temp, y_temp)
    balanced_data = pd.concat([X_balanced, y_balanced], axis=1)

    if extra > 0:# Add extra rows if specified
        balanced_data = pd.concat([balanced_data, rejected_false], axis=0)

    print(f"Total Data: {balanced_data.shape[0]}")
    
    return original_data, balanced_data, target_col

def setup_preprocessing():
    os.makedirs('models/classifiers', exist_ok=True)
    os.makedirs('failed_predictions', exist_ok=True)

def prepare_features(balanced_data):
    categorical_cols = [
        'STATUS', 'ITNCHGCOMCANAL', 'ITNESCENARIO',
        'ACTSTART', 'PMCHGTYPE', 'OWNERGROUP', 'PMCHGCONCERN',
        'ITDCLOSURECODE', 'PROBLEMCODE', 'ITDCHCREATEDBYGROUP',
        'ACTFINISH', 'FIRSTAPPRSTATUS', 'PMCHGAPPROVALSTATE',
        'ITNAMBITO'
    ]

    text_cols = ['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'REASONFORCHANGE_trad']
    numeric_cols = ['DESCRIPTION_len', 'ITNDESCIMPACT_len', 'REASONFORCHANGE_len']

    # Calcular las longitudes de las columnas de texto y guardarlas en numeric_cols
    for text_col, numeric_col in zip(text_cols, numeric_cols):
        if text_col in balanced_data.columns:
            balanced_data[numeric_col] = balanced_data[text_col].fillna("").astype(str).str.len()

    # Fill missing values
    for col in categorical_cols + text_cols:
        if col in balanced_data.columns:
            balanced_data[col] = balanced_data[col].fillna('')

    # Text embeddings
    print("\nGenerating features...")

    # Preparar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

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

    # Concatenar embeddings o crear array vacío si no hay columnas válidas
    all_text_embeddings = np.hstack(text_embeddings) if text_embeddings else np.zeros((len(balanced_data), 0))

    # Save embeddings
    with open('/fhome/amir/TFG/code/classificators/DL/embeddings/text_embeddings.pkl', 'wb') as f:
        pickle.dump(all_text_embeddings, f)
    print("Embeddings saved to text_embeddings.pkl")

    # # Cargar embeddings
    # with open('/fhome/amir/TFG/code/classificators/DL/embeddings/text_embeddings.pkl', 'rb') as f:
    #     all_text_embeddings = pickle.load(f)
    # print("Embeddings loaded from text_embeddings.pkl")

    # Categorical encoding
    label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_categorical = label_encoder.fit_transform(balanced_data[categorical_cols])

    # Combine features
    X = hstack([
        csr_matrix(all_text_embeddings.astype(np.float32)), 
        csr_matrix(encoded_categorical.astype(np.float32)),
        csr_matrix(balanced_data[numeric_cols].astype(np.float32).values)
    ])
    y = balanced_data['REJECTED'].values
    
    return X, y, categorical_cols, text_cols

def evaluate_model(model, loader, criterion, device, return_probs=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = outputs.squeeze()
            predicted = (probs > 0.5).float()

            running_loss += loss.item() * inputs.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total

    if return_probs:
        return epoch_loss, epoch_acc, np.array(all_probs), np.array(all_labels)
    else:
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)



def evaluate_and_save_best_model_outputs(best_model, test_dataset, y_test, balanced_data, original_data, class_names, target_col, name_model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = nn.BCELoss()

    _, _, y_best_probs, y_best_true = evaluate_model(best_model, best_test_loader, criterion, device, return_probs=True)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_best_true, y_best_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    wandb.log({"Best_Model_ROC_Curve": wandb.Image(plt)})
    plt.close()

    # Confusion Matrix
    y_best_labels = (y_best_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_best_true, y_best_labels)
    plot_confusion_matrix(cm, 'Best Model Confusion Matrix', class_names)
    wandb.log({"Best_Model_Confusion_Matrix": wandb.Image(plt)})

    # Guardar el modelo
    model_path = f'models/classifiers/{name_model}_bm.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'input_size': best_model[0].in_features if isinstance(best_model, nn.Sequential) else None
    }, model_path)
    wandb.save(model_path)

    print("Evaluación y guardado del mejor modelo completado.")

    return model_path

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Convertir el DataLoader en una lista y hacer shuffle manualmente
    shuffled_loader = list(train_loader)
    np.random.shuffle(shuffled_loader)

    for inputs, labels in shuffled_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def single_training_loop(X, y, args, balanced_data, original_data, nepoch):
    class_names = ['False', 'True']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # Count per balance classes
    class_counts = Counter(y_train)
    neg = class_counts[0]
    pos = class_counts[1]
    pos_weight = torch.tensor([neg / pos], dtype=torch.float)

    # Create Dataset
    train_dataset = CHDataset(X_train, y_train)
    test_dataset = CHDataset(X_test, y_test)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TabNetModel(X.shape[1]).to(device)

    # Init weights
    model.apply(init_weights)

    criterion = nn.BCELoss()  # output sin promedio para aplicar weights manuales
    optimizer = optim.Adam(model.parameters())
    best_val_loss = float('inf')

    # Definir el scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Training loop
    for epoch in range(nepoch):
        # 1. train epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 2. test epoch
        test_loss_epoch, test_acc_epoch, _, _ = evaluate_model(model, test_loader, criterion, device)
    
        # 3. Ajustar el learning rate usando el scheduler basado en el val_loss
        scheduler.step(test_loss_epoch)

        wandb.log({
            f"epoch": epoch,
            f"train_loss": train_loss,
            f"test_loss": test_loss_epoch,
            f"train_acc": train_acc,
            f"test_acc": test_acc_epoch
        })

    # Evaluate on test set
    test_loss, test_acc, y_test_pred, y_test_true = evaluate_model(model, test_loader, criterion, device)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_true, y_test_pred),
        'precision_macro': precision_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    }

    # Store metrics
    for name, value in metrics.items():
        wandb.log({f"{name}": value})

    # Recall por clase
    for class_idx, class_name in enumerate(class_names):
        recall = recall_score(y_test_true == class_idx, y_test_pred == class_idx, zero_division=0)
        wandb.log({f"recall_{class_name}": recall})
        metrics[f'recall_{class_name}'] = recall

    # Confusion matrix
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix', class_names)

    print(f"\nClassification Report")
    print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))

    return metrics, model, test_dataset, y_test


from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch

def train_tabnet_model(X_train, y_train, X_valid, y_valid, X_test, y_test, num_epochs=100, batch_size=1024):
    # Crear el clasificador TabNet
    clf = TabNetClassifier()

    # Ajustar el modelo
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        max_epochs=num_epochs,
        batch_size=batch_size,
        virtual_batch_size=128,
        patience=10,
        # Additional parameters can be adjusted based on the dataset and task
    )

    # Realizar predicciones en el conjunto de test
    preds = clf.predict(X_test)
    
    # Evaluar las métricas
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision_macro': precision_score(y_test, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, preds, average='macro', zero_division=0)
    }
    
    for name, value in metrics.items():
        wandb.log({f"{name}": value})

    # Confusion matrix
    cm_test = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm_test)

    return metrics, clf@

def plot_confusion_matrix(cm, class_names=['False', 'True']):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    wandb.log({"Confusion_Matrix": wandb.Image(fig)})
    plt.close()



def save_single_metrics(metrics, class_names, name_model):

    # Extraer métricas generales
    summary_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision (Macro)': metrics['precision_macro'],
        'Recall (Macro)': metrics['recall_macro'],
        'F1-score (Macro)': metrics['f1_macro'],
        'Recall (True)': metrics['recall_True'],
        'Recall (False)': metrics['recall_False'],
    }

    if 'rpa_recall' in metrics and 'human_recall' in metrics:
        summary_metrics['RPA Recall'] = metrics['rpa_recall']
        summary_metrics['Human Recall'] = metrics['human_recall']

    # Log a W&B
    wandb.log(summary_metrics)

    # Guardar CSV
    csv_path = f'/fhome/amir/TFG/code/classificators/DL/results/{name_model}_single_metrics.csv'
    headers = ['Metric', 'Value']
    rows = [[metric, f"{value:.4f}"] for metric, value in summary_metrics.items()]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    # Log W&B Table
    df_metrics = pd.DataFrame(rows, columns=headers)
    wandb.log({f"{name_model}_metrics_table": wandb.Table(dataframe=df_metrics)})

    # Plot RPA vs HUMAN recall si aplica
    if 'rpa_recall' in metrics and 'human_recall' in metrics:
        plt.figure()
        plt.bar(['RPA', 'HUMAN'], [metrics['rpa_recall'], metrics['human_recall']], color=['orange', 'blue'])
        plt.title('Recall por tipo de autor')
        plt.ylabel('Recall')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        wandb.log({"Recall por tipo de autor": wandb.Image(plt)})
        plt.close()

    return csv_path


def main():
    ## ------------- 1. Initialize WandB --------------------##
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'Tabnet1_HUMAN'
    args = parse_arguments()
    nepoch = 150
    extra = 0
    initialize_wandb(args, name, nepoch, extra)

    ## ------------- 2. Load and preprocess data --------------------##
    sys.stderr.write(" -------------------- 2. Load and preprocess data --------------------\n")
    original_data, balanced_data, target_col = load_and_balance_data(extra)
    setup_preprocessing()

    # # sample balanced data
    # balanced_data = balanced_data.sample(n=1000, random_state=42)

    X, y, categorical_cols, text_cols = prepare_features(balanced_data)
    
    class_names = ['False', 'True']
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 3. Training and Testing--------------------##
    sys.stderr.write(" -------------------- 3. Training  ----------------------------------\n")
    metrics, model, test_dataset, y_test = train_tabnet_model(X, y, args, balanced_data, original_data, nepoch)
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 4. Save metrics and predictions --------------------##
    sys.stderr.write(" -------------------- 4. Save metrics and predictions --------------------\n")
    model_path = evaluate_and_save_best_model_outputs(model, test_dataset, pd.Series(y_test), balanced_data, original_data, class_names, target_col, name)

    csv_path = save_single_metrics(metrics, class_names, name)

    # save code
    code_path = '/fhome/amir/TFG/code/classificators/DL/Tabnet_HUMANonly.py'
    wandb.save(code_path)

    sys.stderr.write('----------------------------------------------------------------------\n')
    
    ## ------------- 5. Finalize WandB --------------------##
    sys.stderr.write(" -------------------- 5. Finalize WandB --------------------\n")
    wandb.finish()
    sys.stderr.write(f"\nBest model saved to {model_path}\n")
    sys.stderr.write(f"Metrics saved to {csv_path}\n")
    if args.save_fails:
        sys.stderr.write(f"Failed predictions saved to failed_predictions/ directory\n")
    sys.stderr.write("All done!\n")
    sys.stderr.write('----------------------------------------------------------------------\n')

if __name__ == "__main__":
    main()
