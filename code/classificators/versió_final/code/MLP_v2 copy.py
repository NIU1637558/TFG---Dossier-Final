import os
import pandas as pd
import numpy as np
import csv
import argparse
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                            f1_score, classification_report, confusion_matrix)
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
from model_arquitectres.models import AttentionMLP

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

def evaluate_and_save_best_model_outputs(best_fold, X, y, balanced_data, original_data, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model = AttentionMLP(best_fold['input_size']).to(device)
    best_model.load_state_dict(best_fold['model_state'])
    best_model.eval()

    best_test_dataset = CHDataset(X[best_fold['test_idx']], y[best_fold['test_idx']])
    best_test_loader = DataLoader(best_test_dataset, batch_size=64, shuffle=False)

    criterion = nn.BCELoss()
    _, _, y_best_pred, y_best_true = evaluate_model(best_model, best_test_loader, criterion, device, return_probs=True)

    # ROC Curve
    y_best_prob = np.array(y_best_pred)
    fpr, tpr, _ = roc_curve(y_best_true, y_best_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
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
    y_best_labels = (y_best_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_best_true, y_best_labels)

    fig_cm = plot_confusion_matrix(cm, 'Best Model Confusion Matrix', class_names, return_fig=True)
    wandb.log({"Best_Model_Confusion_Matrix": wandb.Image(fig_cm)})
    plt.close(fig_cm)

    # Distribución RPA/HUMAN
    original_indices = balanced_data.iloc[best_fold['test_idx']].index
    subset_original = original_data.loc[original_indices].copy()
    subset_original['PREDICTED'] = y_best_labels

    if 'author' in subset_original.columns:
        subset_original['author_type'] = subset_original['author'].apply(
            lambda x: 'RPA' if 'RPA' in str(x).upper() else 'HUMAN'
        )
        author_counts = subset_original.groupby(['author_type', 'PREDICTED']).size().unstack(fill_value=0)

        author_counts.plot(kind='bar', stacked=True)
        plt.title('Distribución de Predicciones por Tipo de Autor - Best Model')
        plt.ylabel('Número de Predicciones')
        plt.xlabel('Tipo de Autor')
        plt.tight_layout()
        wandb.log({"Best_Model_Author_Distribution": wandb.Image(plt)})
        plt.close()

    wandb.log({"Best_Model_ROC_AUC": roc_auc})
    print("Evaluación del mejor modelo finalizada y resultados subidos a wandb.")


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
    
    # Guardar registros fallidos del mejor fold con WONUM
    if args.save_fails and failed_predictions:
        all_failed = pd.concat(failed_predictions)
        best_fold_failed = all_failed[all_failed['FOLD'] == best_fold['fold']].copy()
        
        if 'WONUM' in best_fold_failed.columns:
            failed_wonums = best_fold_failed['WONUM'].unique()
            pd.DataFrame({'WONUM': failed_wonums}).to_csv(
                '/fhome/amir/TFG/code/classificators/DL/failed_predictions/failed_wonums.csv', index=False)
        else:
            print("WONUM column not found in failed predictions.")
    
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

def save_metrics(fold_metrics, n_splits, class_names, name_model):
    overall_metrics = {
        'accuracy': np.mean(fold_metrics['overall_accuracy']),
        'precision_macro': np.mean(fold_metrics['overall_precision_macro']),
        'recall_macro': np.mean(fold_metrics['overall_recall_macro']),
        'f1_macro': np.mean(fold_metrics['overall_f1_macro']),
        'recall_True': np.mean(fold_metrics['recall_class_True']),
        'recall_False': np.mean(fold_metrics['recall_class_False'])
    }

    print("\nOverall Metrics:")
    for name, value in overall_metrics.items():
        print(f"{name}: {value:.4f}")

    wandb.log({
        'overall_accuracy': overall_metrics['accuracy'],
        'overall_precision_macro': overall_metrics['precision_macro'],
        'overall_recall_macro': overall_metrics['recall_macro'],
        'overall_f1_macro': overall_metrics['f1_macro'],
        'recall_True': overall_metrics['recall_True'],
        'recall_False': overall_metrics['recall_False']
    })

    # Save metrics
    csv_path = f'/fhome/amir/TFG/code/classificators/DL/results/{name_model}_metrics.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['Metric', 'Average', 'Std Dev'] + [f'Fold {i+1}' for i in range(n_splits)])
        
        for class_name in class_names:
            recalls = fold_metrics[f'recall_class_{class_name}']
            writer.writerow([
                f"Recall ({class_name})",
                f"{np.mean(recalls):.4f}",
                f"{np.std(recalls):.4f}"
            ] + [f"{v:.4f}" for v in recalls])
        
        global_metrics = {
            'Accuracy': 'overall_accuracy',
            'Precision': 'overall_precision_macro',
            'F1-score': 'overall_f1_macro'
        }
        
        for name, key in global_metrics.items():
            values = fold_metrics[key]
            writer.writerow([
                name,
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}"
            ] + [f"{v:.4f}" for v in values])
        
        writer.writerow([
            "Recall (Macro)",
            f"{np.mean(fold_metrics['overall_recall_macro']):.4f}",
            f"{np.std(fold_metrics['overall_recall_macro']):.4f}"
        ] + [f"{v:.4f}" for v in fold_metrics['overall_recall_macro']])
    
    return csv_path

def save_predictions_and_model(best_fold, X, y, balanced_data, original_data, target_col, name_model):
    # Recreate model and load best state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionMLP(best_fold['input_size']).to(device)
    model.load_state_dict(best_fold['model_state'])
    model.eval()
    
    # Create dataset and loader for best fold test set
    test_dataset = CHDataset(X[best_fold['test_idx']], y[best_fold['test_idx']])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds.flatten())
    
    # Save predictions
    original_indices = balanced_data.iloc[best_fold['test_idx']].index
    results_df = original_data.loc[original_indices].copy()
    results_df['PREDICTED'] = np.array(all_preds)
    results_df['CORRECT'] = (results_df[target_col] == results_df['PREDICTED'])

    results_path = f'models/classifiers/{name_model}_predicts.csv'
    results_df.to_csv(results_path, index=False)

    # Save the best model
    model_path = f'models/classifiers/{name_model}_bm.pth'
    torch.save({
        'model_state_dict': best_fold['model_state'],
        'input_size': best_fold['input_size']
    }, model_path)
    
    return results_path, model_path

def main():
    ## ------------- 1. Initialize WandB --------------------##
    name = 'MLP_v2_att1'
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
    evaluate_and_save_best_model_outputs(
    best_fold=best_fold,
    X=X,
    y=y,
    balanced_data=balanced_data,
    original_data=original_data,
    class_names=class_names
)

    csv_path = save_metrics(fold_metrics, n_splits, class_names, name)
    results_path, model_path = save_predictions_and_model(
        best_fold, X, y, balanced_data, original_data, target_col, name)
    
    ## ------------- 5. Finalize WandB --------------------##
    wandb.finish()
    print(f"\nBest model saved to {model_path}")
    print(f"Metrics saved to {csv_path}")
    print(f"Complete predictions saved to {results_path}")
    if args.save_fails:
        print(f"Failed predictions saved to failed_predictions/ directory")

if __name__ == "__main__":
    main()