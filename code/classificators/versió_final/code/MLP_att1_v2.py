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

# import model
from model_arquitectres.models import MLP, AttentionMLP

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
        'STATUS', 'ITNCHGCOMCANAL', 'REASONFORCHANGE', 'ITNESCENARIO',
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
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to('cuda' if torch.cuda.is_available() else 'cpu')
    text_embeddings = [model.encode(balanced_data[col].tolist(), show_progress_bar=True, device=model.device) 
                      for col in text_cols if col in balanced_data.columns]
    all_text_embeddings = np.concatenate(text_embeddings, axis=1) if text_embeddings else np.zeros((len(balanced_data), 0))

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

    for inputs, labels in train_loader:
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



def evaluate_and_save_best_model_outputs(best_fold, X, y, balanced_data, original_data, class_names, target_col, name_model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model = AttentionMLP(best_fold['input_size']).to(device)
    best_model.load_state_dict(best_fold['model_state'])
    best_model.eval()

    best_test_dataset = CHDataset(X[best_fold['test_idx']], y[best_fold['test_idx']])
    best_test_loader = DataLoader(best_test_dataset, batch_size=64, shuffle=False)

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
    original_indices = balanced_data.iloc[best_fold['test_idx']].index
    subset_original = original_data.loc[original_indices].copy()
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
        'model_state_dict': best_fold['model_state'],
        'input_size': best_fold['input_size']
    }, model_path)
    wandb.save(model_path)

    print("Evaluación y guardado del mejor modelo completado.")

    return model_path


def cross_validation_loop(X, y, n_splits, args, balanced_data, original_data):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    class_names = ['False', 'True']
    fold_metrics = defaultdict(list)
    best_fold = {'loss': float('inf'), 'model_state': None, 'fold': -1, 'val_score': -1, 'test_idx': None}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.save_fails:
        failed_predictions = []

    print('X shape:', X.shape)
    print('y shape:', y.shape)

    print(f"\nStarting {n_splits}-fold cross validation...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n=== Fold {fold} ===")
        
        # Split train into train and validation (80% train, 20% validation)
        train_idx, val_idx = next(StratifiedKFold(n_splits = 5, shuffle=False).split(X[train_idx], y[train_idx]))
        
        # Create datasets
        train_dataset = CHDataset(X[train_idx], y[train_idx])
        val_dataset = CHDataset(X[val_idx], y[val_idx])
        test_dataset = CHDataset(X[test_idx], y[test_idx])

        print('datasets shapes'
              f'\ntrain: {X[train_idx].shape}, val: {X[val_idx].shape}, test: {X[test_idx].shape}')
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize model, criterion and optimizer
        model = AttentionMLP(X.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        for epoch in range(50):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
            
            wandb.log({
                f"fold{fold}_epoch": epoch,
                f"fold{fold}_train_loss": train_loss,
                f"fold{fold}_val_loss": val_loss,
                f"fold{fold}_train_acc": train_acc,
                f"fold{fold}_val_acc": val_acc,
                "global_epoch": epoch
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Evaluate on validation set
            test_loss_epoch, test_acc_epoch, _, _ = evaluate_model(model, test_loader, criterion, device)
            
            # WandB log con test
            wandb.log({
                f"fold{fold}_epoch": epoch,
                f"fold{fold}_train_loss": train_loss,
                f"fold{fold}_val_loss": val_loss,
                f"fold{fold}_test_loss": test_loss_epoch,
                f"fold{fold}_train_acc": train_acc,
                f"fold{fold}_val_acc": val_acc,
                f"fold{fold}_test_acc": test_acc_epoch,
                "global_epoch": epoch
            })
        
        # Load best model
        model.load_state_dict(best_model_state)
        
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
            fold_metrics[f'overall_{name}'].append(value)
            wandb.log({f"fold{fold}_{name}": value})
        
        # Recall por clase
        for class_idx, class_name in enumerate(class_names):
            recall = recall_score(y_test_true == class_idx, y_test_pred == class_idx, zero_division=0)
            fold_metrics[f'recall_class_{class_name}'].append(recall)
            wandb.log({f"fold{fold}_recall_{class_name}": recall})
        
        # Confusion matrix
        cm_test = confusion_matrix(y_test_true, y_test_pred)
        plot_confusion_matrix(cm_test, f'Fold {fold} Test Confusion Matrix', class_names)
        
        # Guardar registros fallidos si está activado
        print(args.save_fails)
        if args.save_fails:
            original_indices = balanced_data.iloc[test_idx].index
            failed_mask = (y_test_true != y_test_pred)
            print(failed_mask.shape)
            print(original_data.shape)
            failed_records = original_data.loc[original_indices[failed_mask]].copy()
            failed_records['PREDICTED'] = y_test_pred[failed_mask]
            failed_records['FOLD'] = fold
            
            if args.save_fails:
                failed_predictions.append(failed_records)
        
        print(f"\nClassification Report for Fold {fold}:")
        print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))
        
        # Track best model
        if test_loss < best_fold['loss']:
            best_fold = {
                'loss': test_loss,
                'model_state': model.state_dict(),
                'fold': fold,
                'val_score': test_acc,
                'test_idx': test_idx,
                'input_size': X.shape[1]
            }
    
    return fold_metrics, best_fold


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

def save_metrics(fold_metrics, n_splits, class_names, name_model):
    overall_metrics = {
        'accuracy': np.mean(fold_metrics['overall_accuracy']),
        'precision_macro': np.mean(fold_metrics['overall_precision_macro']),
        'recall_macro': np.mean(fold_metrics['overall_recall_macro']),
        'f1_macro': np.mean(fold_metrics['overall_f1_macro']),
        'recall_True': np.mean(fold_metrics['recall_class_True']),
        'recall_False': np.mean(fold_metrics['recall_class_False']),
    }

    if 'rpa_recall' in fold_metrics and 'human_recall' in fold_metrics:
        overall_metrics['rpa_recall'] = np.mean(fold_metrics['rpa_recall'])
        overall_metrics['human_recall'] = np.mean(fold_metrics['human_recall'])

    wandb.log(overall_metrics)

    # Guardar en CSV
    csv_path = f'/fhome/amir/TFG/code/classificators/DL/results/{name_model}_metrics.csv'
    rows = []
    headers = ['Metric', 'Average', 'Std Dev', 'Avg ± Std'] + [f'Fold {i+1}' for i in range(n_splits)]

    def fmt(value):
        return f"{value:.4f}"

    def avg_std_row(name, values):
        avg = np.mean(values)
        std = np.std(values)
        return [name, fmt(avg), fmt(std), f"{fmt(avg)} ± {fmt(std)}"] + [fmt(v) for v in values]

    for class_name in class_names:
        recalls = fold_metrics[f'recall_class_{class_name}']
        rows.append(avg_std_row(f"Recall ({class_name})", recalls))

    global_metrics = {
        'Accuracy': 'overall_accuracy',
        'Precision': 'overall_precision_macro',
        'F1-score': 'overall_f1_macro',
        'Recall (Macro)': 'overall_recall_macro'
    }

    for name, key in global_metrics.items():
        values = fold_metrics[key]
        rows.append(avg_std_row(name, values))

    if 'rpa_recall' in fold_metrics and 'human_recall' in fold_metrics:
        for label in ['rpa_recall', 'human_recall']:
            values = fold_metrics[label]
            label_title = label.replace('_', ' ').title()
            rows.append(avg_std_row(label_title, values))

    # Guardar CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    # Log Table to W&B
    df_metrics = pd.DataFrame(rows, columns=headers)
    wandb.log({f"{name_model}_metrics_table": wandb.Table(dataframe=df_metrics)})

    # Gráfico RPA vs HUMAN si aplica
    if 'rpa_recall' in fold_metrics and 'human_recall' in fold_metrics:
        plt.figure()
        plt.plot(fold_metrics['rpa_recall'], label='RPA Recall')
        plt.plot(fold_metrics['human_recall'], label='HUMAN Recall')
        plt.xlabel('Fold')
        plt.ylabel('Recall')
        plt.title('Recall por tipo de autor (RPA vs HUMAN)')
        plt.legend()
        wandb.log({"Recall por tipo de autor": wandb.Image(plt)})
        plt.close()

    return csv_path

def main():
    ## ------------- 1. Initialize WandB --------------------##
    name = 'MLP_att1_v2'
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
    fold_metrics, best_fold = cross_validation_loop(X, y, n_splits, args, balanced_data, original_data)
    
    ## ------------- 4. Save metrics and predictions --------------------##
    model_path = evaluate_and_save_best_model_outputs(
    best_fold=best_fold,
    X=X,
    y=y,    
    balanced_data=balanced_data,
    original_data=original_data,
    class_names=class_names,
    target_col=target_col,
    name_model=name
    )


    csv_path = save_metrics(fold_metrics, n_splits, class_names, name)
    
    ## ------------- 5. Finalize WandB --------------------##
    wandb.finish()
    print(f"\nBest model saved to {model_path}")
    print(f"Metrics saved to {csv_path}")
    if args.save_fails:
        print(f"Failed predictions saved to failed_predictions/ directory")

if __name__ == "__main__":
    main()