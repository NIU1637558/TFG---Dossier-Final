import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                            f1_score, classification_report, confusion_matrix)
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
import torch
import pickle
import wandb
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
import seaborn as sns

class MLPExperiment:
    def __init__(self, name, train_condition, test_condition, save_fails=False):
        self.name = name
        self.train_condition = train_condition
        self.test_condition = test_condition
        self.save_fails = save_fails
        self.categorical_cols = [
            'STATUS', 'ITNCHGCOMCANAL', 'REASONFORCHANGE', 'ITNESCENARIO',
            'ACTSTART', 'PMCHGTYPE', 'OWNERGROUP', 'PMCHGCONCERN',
            'ITDCLOSURECODE', 'PROBLEMCODE', 'ITDCHCREATEDBYGROUP',
            'ACTFINISH', 'FIRSTAPPRSTATUS', 'PMCHGAPPROVALSTATE',
            'ITNAMBITO', 'ITNMOTIVO_WL'
        ]
        self.text_cols = ['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'DESCRIPT_WL', 'REASONFORCHANGE_trad']
        self.target_col = 'REJECTED'
        self.class_names = ['False', 'True']
        
    def load_and_split_data(self, data_path):
        """Load data and split ensuring no overlap between train and test"""
        print(f"\nLoading and splitting data for {self.name}...")
        data = pd.read_csv(data_path)
        
        # Separate REJECTED False (no author) and True (with author)
        false_data = data[data[self.target_col] == False]
        true_data = data[data[self.target_col] == True]
        
        # Split true data into RPA and HUMAN
        rpa_data = true_data[true_data['author'] == 'RPA']
        human_data = true_data[true_data['author'] == 'HUMAN']
        
        # Split ALL data (both false and true) into train and test FIRST to ensure no overlap
        false_train, false_test = train_test_split(false_data, test_size=0.2, random_state=42)
        
        # Split true data according to author type
        if self.train_condition == "rpa_data":
            train_true, test_true = train_test_split(rpa_data, test_size=0.2, random_state=42)
        elif self.train_condition == "human_data":
            train_true, test_true = train_test_split(human_data, test_size=0.2, random_state=42)
        else:
            raise ValueError(f"Invalid train condition: {self.train_condition}")
        
        if self.name != "exp3_HUMANonly" and self.name != "exp4_RPAonly":
            # For test data, use the opposite condition
            if self.test_condition == "human_data":
                test_true = human_data[~human_data.index.isin(train_true.index)] if self.train_condition == "rpa_data" else human_data
            elif self.test_condition == "rpa_data":
                test_true = rpa_data[~rpa_data.index.isin(train_true.index)] if self.train_condition == "human_data" else rpa_data
            else:
                raise ValueError(f"Invalid test condition: {self.test_condition}")
        
        # Balance train set (undersample False to match True count)
        false_train = false_train.sample(n=len(train_true), random_state=42)
        train_data = pd.concat([false_train, train_true])
        
        # Balance test set (undersample False to match True count)
        false_test = false_test.sample(n=len(test_true), random_state=42)
        test_data = pd.concat([false_test, test_true])
        
        # Verify no overlap
        train_indices = set(train_data.index)
        test_indices = set(test_data.index)
        assert len(train_indices.intersection(test_indices)) == 0, "Train and test sets overlap!"
        
        print(f"Train size: {len(train_data)} (False: {len(false_train)}, True: {len(train_true)})")
        print(f"Test size: {len(test_data)} (False: {len(false_test)}, True: {len(test_true)})")
        
        return train_data, test_data, data


    def preprocess_data(self, data):
        """Preprocess data: fill NAs and prepare structures"""
        for col in self.categorical_cols + self.text_cols:
            if col in data.columns:
                data[col] = data[col].fillna('')
        return data
    
    def generate_features(self, data):
        """Generate features: text embeddings and categorical encoding"""
        print(f"\nGenerating features for {self.name}...")
        
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        text_embeddings = [
            model.encode(data[col].tolist(), show_progress_bar=True, device=model.device) 
            for col in self.text_cols if col in data.columns
        ]
        
        all_text_embeddings = np.concatenate(text_embeddings, axis=1) if text_embeddings else np.zeros((len(data), 0))
        
        label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded_categorical = label_encoder.fit_transform(data[self.categorical_cols])
        
        X = hstack([
            csr_matrix(all_text_embeddings.astype(np.float32)), 
            csr_matrix(encoded_categorical.astype(np.float32))
        ])
        y = data[self.target_col].values
        
        return X, y, label_encoder
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, test_data, original_data):
        """Train and evaluate MLP model"""
        print(f"\nTraining and evaluating for {self.name}...")
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.25
        )
        mlp.fit(X_train, y_train)
        
        y_train_pred = mlp.predict(X_train)
        y_test_pred = mlp.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision_macro': precision_score(y_test, y_test_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_test_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_test_pred, average='macro', zero_division=0),
            'recall_True': recall_score(y_test == True, y_test_pred == True, zero_division=0),
            'recall_False': recall_score(y_test == False, y_test_pred == False, zero_division=0)
        }
        
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        
        for epoch, loss in enumerate(mlp.loss_curve_):
            wandb.log({
                f"{self.name}_epoch": epoch,
                f"{self.name}_train_loss": loss,
                "global_epoch": epoch
            })
        
        failed_predictions = None
        if self.save_fails:
            failed_mask = (y_test != y_test_pred)
            failed_records = test_data[failed_mask].copy()
            failed_records['PREDICTED'] = y_test_pred[failed_mask]
            failed_records['EXP_NAME'] = self.name
            original_indices = failed_records.index
            failed_predictions = original_data.loc[original_indices].copy()
            failed_predictions['PREDICTED'] = failed_records['PREDICTED']
            failed_predictions['EXP_NAME'] = self.name
        
        return mlp, metrics, train_cm, test_cm, failed_predictions
    
    def run_experiment(self, data_path):
        """Run complete experiment pipeline"""
        train_data, test_data, original_data = self.load_and_split_data(data_path)
        train_data = self.preprocess_data(train_data)
        test_data = self.preprocess_data(test_data)
        
        X_train, y_train, _ = self.generate_features(train_data)
        X_test, y_test, _ = self.generate_features(test_data)
        
        mlp, metrics, train_cm, test_cm, failed_predictions = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, test_data, original_data)
        
        for metric_name, value in metrics.items():
            wandb.log({f"{self.name}_{metric_name}": value})
        
        self.plot_confusion_matrix(test_cm, f'{self.name} Test Confusion Matrix')
        
        return metrics, failed_predictions

    def plot_confusion_matrix(self, cm, title):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        wandb.log({title: wandb.Image(plt)})
        plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fails', type=bool, default=False, 
                        help='Guardar registros con predicciones fallidas')
    return parser.parse_args()

def setup_wandb(args):
    """Initialize Weights & Biases configuration"""
    config = {
        "model": "MLP",
        "hidden_layers": "(100, 50)",
        "activation": "relu",
        "solver": "adam",
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.25,
        "dataset": "CH_Total_label2_trad",
        "balance_method": "RandomUnderSampling",
        "save_fails": args.save_fails
    }
    
    wandb.init(
        project="TFG",
        entity="andreu-mir-uab",
        name="MLP_experiments",
        config=config
    )

def main():
    args = parse_arguments()
    setup_wandb(args)
    data_path = '/fhome/amir/TFG/data/CH_Total1.csv'
    
    os.makedirs('failed_predictions', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    experiments = [
        MLPExperiment("exp1_trainRPA_testHUMAN", "rpa_data", "human_data", args.save_fails),
        MLPExperiment("exp2_trainHUMAN_testRPA", "human_data", "rpa_data", args.save_fails),
        MLPExperiment("exp3_HUMANonly", "human_data", "human_data", args.save_fails),
        MLPExperiment("exp4_RPAonly", "rpa_data", "rpa_data", args.save_fails)
    ]
    
    all_metrics = {}
    all_failed_predictions = []
    
    # Crear tabla de wandb para las métricas
    metrics_table = wandb.Table(columns=[
        "Experiment", 
        "Accuracy", 
        "Precision (Macro)", 
        "Recall (Macro)", 
        "F1 (Macro)", 
        "Recall (True)", 
        "Recall (False)"
    ])
    
    for experiment in experiments:
        metrics, failed_predictions = experiment.run_experiment(data_path)
        all_metrics[experiment.name] = metrics
        
        # Añadir fila a la tabla
        metrics_table.add_data(
            experiment.name,
            metrics['accuracy'],
            metrics['precision_macro'],
            metrics['recall_macro'],
            metrics['f1_macro'],
            metrics['recall_True'],
            metrics['recall_False']
        )
        
        if args.save_fails and failed_predictions is not None:
            all_failed_predictions.append(failed_predictions)
    
    # Loggear la tabla completa
    wandb.log({"Metrics Summary": metrics_table})
    
    if args.save_fails and all_failed_predictions:
        all_failed = pd.concat(all_failed_predictions)
        failed_path = 'failed_predictions/MLP_experiments_fails.csv'
        all_failed.to_csv(failed_path, index=False)
        print(f"\nSaved failed predictions to {failed_path}")
    
    # Save metrics to CSV
    csv_path = 'results/MLP_experiments_metrics.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Experiment', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 
                         'F1 (Macro)', 'Recall (True)', 'Recall (False)'])
        for exp_name, metrics in all_metrics.items():
            writer.writerow([
                exp_name,
                metrics['accuracy'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                metrics['f1_macro'],
                metrics['recall_True'],
                metrics['recall_False']
            ])
    
    wandb.finish()
    print(f"\nAll experiments completed. Metrics saved to {csv_path}")

if __name__ == "__main__":
    main()