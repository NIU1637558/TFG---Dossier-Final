# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import load_and_balance_data, load_and_balance_data_all

# 3. PREPROCESS DATA
from functions.preprocess_data import prepare_features, install_nltk_resources, clean_stopwords

# 4. TRAIN/EVALUATION
from functions.training import single_training_loop, evaluate_and_save_best_model_outputs, save_single_metrics

# 5. MODELS
from model_arquitectres.models import MLP2, HybridAttentionMLP

# others
import pandas as pd 
import numpy as np
import sys
import wandb
import os

from functions.training import bagging_training_loop  # Nueva función que implementaremos

def run_experiment(embedder, data_mode, name_prefix):
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")

    if data_mode not in ['cat', 'txt', 'all']:
        raise ValueError("data_mode debe ser uno de: 'cat', 'txt', 'all'")

    name = f"{name_prefix}_{data_mode}"
    args = parse_arguments()
    nepoch = 150
    extra = 0
    initialize_wandb(args, name, nepoch, extra)

    # --------- 2. Load and preprocess data ---------
    if 'RPA' in name:
        data_type = 'RPA'
    elif 'HUMAN' in name:
        data_type = 'human'
    else:
        data_type = 'ALL'
    print('data_type:', data_type)
    if data_type == 'ALL':
        original_data, balanced_data, target_col = load_and_balance_data_all(extra, dq = True)
    else:
        print('loading with authors:', data_type)
        original_data, balanced_data, target_col = load_and_balance_data(extra,data_type)
    setup_preprocessing()
    # Filtrar stopwords
    text_cols = ['DESCRIPTION_trad', 'REASONFORCHANGE_trad']

    install_nltk_resources()
    for col in text_cols:
        balanced_data[col] = balanced_data[col].apply(clean_stopwords)
        original_data[col] = original_data[col].apply(clean_stopwords)

    # Preparar features
    X, y, categorical_cols, text_cols, og_indices, txt_shape = prepare_features(
        balanced_data, original_data, embedder=embedder, target_col=target_col
    )

    # Filtrar columnas según el tipo de dato
    if data_mode == 'cat':
        # Eliminar columnas de texto
        balanced_data = balanced_data.drop(columns=text_cols)
        original_data = original_data.drop(columns=text_cols)
        embedder = None  # No se necesita embedding en este caso

    elif data_mode == 'txt':
        # Mantener solo columnas de texto
        balanced_data = balanced_data[text_cols + ['original_index', target_col]]
        original_data = original_data[text_cols + ['original_index', target_col]]

    if embedder is None and data_mode != 'cat':
        raise ValueError("Debe proporcionarse un embedder para modos 'txt' o 'all'")

    # Entrenamiento Bagging
    att = 'att' in name
    n_estimators = 20
    sample_ratio = 1
    
    print("Balanced data target count:\n", balanced_data[target_col].value_counts())
    print("Y distribution:", np.unique(y, return_counts=True))

    metrics, ensemble, test_loader, y_test = bagging_training_loop(
        X, y, args, balanced_data, original_data, nepoch,
        n_estimators=n_estimators, sample_ratio=sample_ratio,
        att=att, txt_shape=txt_shape
    )

    # Guardar resultados
    model_path = evaluate_and_save_best_model_outputs(
        ensemble, test_loader, y_test, balanced_data,
        original_data, ['False', 'True'], target_col, name, False, txt_shape
    )

    csv_path = save_single_metrics(metrics, ['False', 'True'], name)

    wandb.save('/fhome/amir/TFG/code/classificators/DL/DataQuality_classif2.py')
    wandb.finish()
    sys.stderr.write(f"\n{name}: Model saved to {model_path}, metrics to {csv_path}\n")

    return metrics

if __name__ == "__main__":
    # Lista de embedders válidos
    embedders = ['AE1BERTtuned']

    # Tipos de datos
    data_modes = ['cat','txt','all']

    # Nombre base común del experimento
    author = 'ALL'
    name_prefix_base = f'DQ1_{author}_BAGGING_esc'

    # Diccionario para almacenar métricas por tipo
    results_dict = {dm:[] for dm in data_modes}

    for data_mode in data_modes:
        if data_mode == 'cat':
            # Solo categóricos: sin embedders
            print(f"Running experiment with data_mode: {data_mode} (no embedders)")
            metrics = run_experiment(embedder=None, data_mode='cat', name_prefix=name_prefix_base)
            print(metrics)
            # select only metrics of interesty
            metrics_exp = [
                metrics['recall_True'], 
                metrics['recall_False'], 
                metrics['precision_macro']
            ]
            results_dict['cat'].append(['-', *metrics_exp])
        else:
            for emb in embedders:
                print(f"Running experiment with data_mode: {data_mode} and embedder: {emb}")
                name_prefix = f"{name_prefix_base}_{emb}"
                metrics = run_experiment(embedder=emb, data_mode=data_mode, name_prefix=name_prefix)
                metrics_exp = [
                    metrics['recall_True'], 
                    metrics['recall_False'], 
                    metrics['precision_macro']
                ]
                results_dict[data_mode].append([emb, *metrics_exp])

    # Guardar resultados en Excel
    root_dir = '/fhome/amir/TFG/results/DataQuality2'
    for mode, rows in results_dict.items():
        df = pd.DataFrame(rows, columns=['embedder', 'recT', 'recF', 'prec'])

        if mode == 'cat':
            filename = f'DQcategorical_{author}_imp.csv'
        elif mode == 'txt':
            filename = f'DQtext_only_{author}_imp.csv'
        elif mode == 'all':
            filename = f'DQall_{author}_imp.csv'

        df.to_csv(os.path.join(root_dir, filename), index=False)
    print(f"Results saved to {root_dir}")
    sys.stderr.write("Experiment completed successfully.\n")
    sys.exit(0)
# End of the script 