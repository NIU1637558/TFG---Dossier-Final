import os
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

# import model
from model_arquitectres.models import MLP, AttentionMLP, MLP2

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

def initialize_wandb(args, name_model):
    wandb.init(
        project="TFG",
        entity="andreu-mir-uab",
        name=name_model,
        config={
            "framework": "PyTorch",
            "model": "CustomMLP_v1",
            "hidden_layers": "(100, 50)",
            "activation": "ReLU",
            "optimizer": "Adam",
            "max_epochs": 500,
            "early_stopping": True,
            "patience": 10,
            "n_splits": 5,
            "dataset": "CH_Total_label2_trad",
            "balance_method": "RandomUnderSampling",
            "save_fails": args.save_fails
        }
    )

def load_and_balance_data():
    print("Loading and preprocessing original data...")
    original_data = pd.read_csv('/fhome/amir/TFG/data/CH_Total1.csv')
    target_col = 'REJECTED'

    # Balance data (manteniendo índice original)
    undersampler = RandomUnderSampler()
    X_temp, y_temp = original_data.drop(columns=[target_col]), original_data[target_col]
    X_balanced, y_balanced = undersampler.fit_resample(X_temp, y_temp)
    balanced_data = pd.concat([X_balanced, y_balanced], axis=1)
    
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

    # Categorical encoding
    label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_categorical = label_encoder.fit_transform(balanced_data[categorical_cols])

    # Combine features
    X = hstack([csr_matrix(all_text_embeddings.astype(np.float32)), 
                 csr_matrix(encoded_categorical.astype(np.float32))])
    y = balanced_data['REJECTED'].values
    
    return X, y, categorical_cols, text_cols

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

    # Tabla de distribución RPA/HUMAN para REJECTED True
    subset_original = original_data.loc[y_test.index].copy()
    subset_original['PREDICTED'] = y_best_labels
    subset_original['REAL'] = y_best_true

    if 'author' in subset_original.columns:
        subset_original['author_type'] = subset_original['author'].apply(
            lambda x: 'RPA' if 'RPA' in str(x).upper() else 'HUMAN'
        )

        results = []
        for author_type in ['RPA', 'HUMAN']:
            subset = subset_original[(subset_original['author_type'] == author_type) & (subset_original['REAL'] == 1)]
            total = len(subset)
            correct = (subset['PREDICTED'] == 1).sum()
            results.append({
                'Author Type': author_type,
                'Correct Predicts': correct,
                'Total Registers': total,
                'Ratio': correct / total if total > 0 else 0,
            })

        df_results = pd.DataFrame(results)
        wandb.log({"Best_Model_RPA_HUMAN_Distribution_Table": wandb.Table(dataframe=df_results)})

    # Guardar predicciones fallidas
    failed_mask = subset_original['REAL'] != subset_original['PREDICTED']
    failed_preds = subset_original[failed_mask][['WONUM', 'REAL', 'PREDICTED']]
    failed_table = wandb.Table(dataframe=failed_preds)
    wandb.log({"Best_Model_Failed_Predictions": failed_table})

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


def single_training_loop(X, y, n_splits, args, balanced_data, original_data):
    class_names = ['False', 'True']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    train_dataset = CHDataset(X_train, y_train)
    test_dataset = CHDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    best_val_loss = float('inf')

    # Training loop with early stopping
    patience = 10
    patience_counter = 0
    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        wandb.log({
            f"epoch": epoch,
            f"train_loss": train_loss,
            f"val_loss": val_loss,
            f"train_acc": train_acc,
            f"val_acc": val_acc
        })

        # # Early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model_state = model.state_dict()
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break

        # Evaluate on validation set
        test_loss_epoch, test_acc_epoch, _, _ = evaluate_model(model, test_loader, criterion, device)

        # WandB log con test
        wandb.log({
            f"epoch": epoch,
            f"train_loss": train_loss,
            f"val_loss": val_loss,
            f"test_loss": test_loss_epoch,
            f"train_acc": train_acc,
            f"val_acc": val_acc,
            f"test_acc": test_acc_epoch
        })

    # # Load best model
    # model.load_state_dict(best_model_state)

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

def plot_confusion_matrix(cm, title, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    wandb.log({title: wandb.Image(plt)})
    plt.close()

    return plt.gcf()  # Return the figure object for logging

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
    name = 'MLP_basic2_emb2_v3'
    args = parse_arguments()
    initialize_wandb(args, name)

    ## ------------- 2. Load and preprocess data --------------------##
    original_data, balanced_data, target_col = load_and_balance_data()
    setup_preprocessing()

    # # sample balanced data
    # balanced_data = balanced_data.sample(n=1000, random_state=42)

    X, y, categorical_cols, text_cols = prepare_features(balanced_data)
    
    n_splits = 5
    class_names = ['False', 'True']
    
    ## ------------- 3. Cross-validation and training --------------------##
    metrics, model, test_dataset, y_test = single_training_loop(X, y, n_splits, args, balanced_data, original_data)
    
    ## ------------- 4. Save metrics and predictions --------------------##
    evaluate_and_save_best_model_outputs(model, test_dataset, pd.Series(y_test), balanced_data, original_data, class_names, target_col, name)

    csv_path = save_single_metrics(metrics, class_names, name)
    
    ## ------------- 5. Finalize WandB --------------------##
    wandb.finish()
    print(f"\nBest model saved to {model_path}")
    print(f"Metrics saved to {csv_path}")
    if args.save_fails:
        print(f"Failed predictions saved to failed_predictions/ directory")

if __name__ == "__main__":
    main()