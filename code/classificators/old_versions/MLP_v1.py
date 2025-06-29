import os
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict

# 1. Load the data
dataOG = pd.read_csv('/fhome/amir/TFG/data/CH_Total_label2_trad.csv')

# # data sample
# dataOG = dataOG.sample(1000, random_state=42)

# 2. Preprocess the data
os.makedirs('models', exist_ok=True)

# Columnas categóricas (ajustadas según tu dataset)
categorical_cols = [
    'STATUS', 'ITNCHGCOMCANAL', 'REASONFORCHANGE', 'ITNESCENARIO',
    'ACTSTART', 'PMCHGTYPE', 'OWNERGROUP', 'PMCHGCONCERN',
    'ITDCLOSURECODE', 'PROBLEMCODE', 'ITDCHCREATEDBYGROUP',
    'ACTFINISH', 'FIRSTAPPRSTATUS', 'PMCHGAPPROVALSTATE',
    'LANGCODE', 'ITNAMBITO', 'ENVENTANA', 'ITNMOTIVO_WL'
]

# Columnas de texto libre para embeddings
text_cols = ['DESCRIPTION', 'ITNDESCIMPACT', 'DESCRIPT_WL','REASONFORCHANGE']

# Variable objetivo
target_col = 'REJECTED'

data = dataOG.copy()

# 2.1 EMbeddings 

# Preprocesamiento: asegurar que las columnas categóricas son de tipo correcto
for col in categorical_cols:
    if col in data.columns and data[col].dtype != 'object' and data[col].dtype.name != 'category':
        data[col] = data[col].astype('object')
    # Rellenar NaN con string vacío para columnas categóricas
    if col in data.columns:
        data[col] = data[col].fillna('')

# Rellenar NaN en columnas de texto
for col in text_cols:
    if col in data.columns:
        data[col] = data[col].fillna('')

# Cargar el modelo de Sentence Transformers
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Generar embeddings para cada columna de texto
print("Generando embeddings...")
text_embeddings = []
for col in text_cols:
    if col in data.columns:
        embeddings = model.encode(data[col].tolist(), show_progress_bar=True, device=device)
        text_embeddings.append(embeddings)

# Concatenar todos los embeddings de texto
if text_embeddings:
    all_text_embeddings = np.concatenate(text_embeddings, axis=1)
else:
    all_text_embeddings = np.zeros((len(data), 0))

# Aplicar Label Encoding a las columnas categóricas
print("Aplicando Label Encoding a las columnas categóricas...")
label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoded_categorical = label_encoder.fit_transform(data[categorical_cols])

# Convertir todo a matrices dispersas
text_embeddings_sparse = csr_matrix(all_text_embeddings.astype(np.float32))
encoded_categorical_sparse = csr_matrix(encoded_categorical.astype(np.float32))

# Combinar características
X = hstack([text_embeddings_sparse, encoded_categorical_sparse])
y = data[target_col].values

# Initialize 5-fold cross validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store metrics for each fold
fold_metrics = defaultdict(list)
class_names = ['False', 'True']  # Assuming binary classification with classes 0 and 1

# Perform cross-validation
print(f"\nStarting {n_splits}-fold cross validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n=== Fold {fold} ===")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create and train MLP model
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.25
    )
    
    mlp.fit(X_train, y_train)
    
    # Predictions
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)
    
    # Calculate metrics for each class
    for class_idx, class_name in enumerate(class_names):
        # For binary classification, we can calculate metrics for each class
        y_test_class = (y_test == class_idx)
        y_pred_class = (y_pred == class_idx)
        
        # Store metrics
        fold_metrics[f'accuracy_class_{class_name}'].append(accuracy_score(y_test_class, y_pred_class))
        fold_metrics[f'precision_class_{class_name}'].append(precision_score(y_test_class, y_pred_class, zero_division=0))
        fold_metrics[f'recall_class_{class_name}'].append(recall_score(y_test_class, y_pred_class, zero_division=0))
        fold_metrics[f'f1_class_{class_name}'].append(f1_score(y_test_class, y_pred_class, zero_division=0))
    
    # Also store overall metrics
    fold_metrics['overall_accuracy'].append(accuracy_score(y_test, y_pred))
    fold_metrics['macro_avg_precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
    fold_metrics['macro_avg_recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    fold_metrics['macro_avg_f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
    
    # Print classification report for this fold
    print(f"\nClassification Report for Fold {fold}:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

# Calculate average and std of metrics across folds
results = []
for metric_name, values in fold_metrics.items():
    avg = np.mean(values)
    std = np.std(values)
    results.append({
        'metric': metric_name,
        'avg': avg,
        'std': std,
        'values': values
    })

# Create a detailed report
print("\n=== Cross Validation Results ===")
print("\nClass-specific metrics:")
for class_name in class_names:
    print(f"\nClass {class_name}:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        key = f"{metric}_class_{class_name}"
        avg = np.mean(fold_metrics[key])
        std = np.std(fold_metrics[key])
        print(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")

print("\nOverall metrics:")
for metric in ['overall_accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']:
    avg = np.mean(fold_metrics[metric])
    std = np.std(fold_metrics[metric])
    print(f"{metric}: {avg:.4f} ± {std:.4f}")

# Save metrics to CSV
with open('/fhome/amir/TFG/code/classificators/v1/results/cross_validation_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['Metric', 'Average', 'Std Dev'] + [f'Fold {i+1}' for i in range(n_splits)])
    
    # Write class-specific metrics
    for class_name in class_names:
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            key = f"{metric}_class_{class_name}"
            row = [
                f"{metric.capitalize()} (Class {class_name})",
                f"{np.mean(fold_metrics[key]):.4f}",
                f"{np.std(fold_metrics[key]):.4f}"
            ] + [f"{v:.4f}" for v in fold_metrics[key]]
            writer.writerow(row)
    
    # Write overall metrics
    for metric in ['overall_accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1']:
        row = [
            metric.replace('_', ' ').capitalize(),
            f"{np.mean(fold_metrics[metric]):.4f}",
            f"{np.std(fold_metrics[metric]):.4f}"
        ] + [f"{v:.4f}" for v in fold_metrics[metric]]
        writer.writerow(row)

# Save the model
model_filename = 'models/classifiers/MLP_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(mlp, model_file)

print(f"\nModel saved to {model_filename}")
print("\nMetrics saved to cross_validation_metrics.csv")