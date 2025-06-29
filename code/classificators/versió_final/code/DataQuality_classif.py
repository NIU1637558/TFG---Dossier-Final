# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import *

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



def main():
    ## ------------- 1. Initialize WandB --------------------##
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'DQ_AE1BERTtuned_ALL'
    args = parse_arguments()
    nepoch = 200
    extra = 0
    initialize_wandb(args, name, nepoch, extra)

    ## ------------- 2. Load and preprocess data --------------------##
    sys.stderr.write(" -------------------- 2. Load and preprocess data --------------------\n")
    if 'RPA' in name:
        data_type = 'RPA'
        original_data, balanced_data, target_col = load_and_balance_data(extra, data_type)
    elif 'HUMAN' in name:
        data_type = 'human'
        original_data, balanced_data, target_col = load_and_balance_data(extra, data_type)
    else:
        original_data, balanced_data, target_col = load_and_balance_data_all(extra, dq = True)
    setup_preprocessing()
    print("Balanced data target count aftr pf")
    print(balanced_data[target_col].value_counts())
    print(target_col)
    # # sample balanced data
    # balanced_data = balanced_data.sample(n=1000, random_state=42)
    if 'w2v' in name:
        embedder = 'w2v'
    if 'BERT' in name:
        embedder = 'BERT'
    if 'AE1w2v' in name:
        embedder = 'AE1w2v'
    if 'AE1BERT' in name:
        embedder = 'AE1BERT'
    if 'distBERT' in name:
        embedder = 'distBERT'
    if 'AE1BERTtuned' in name:
        embedder = 'AE1BERTtuned'

    install_nltk_resources()
    # clean stopwords
    balanced_data['DESCRIPTION_trad'] = balanced_data['DESCRIPTION_trad'].apply(clean_stopwords)
    balanced_data['ITNDESCIMPACT_trad'] = balanced_data['ITNDESCIMPACT_trad'].apply(clean_stopwords)
    balanced_data['REASONFORCHANGE_trad'] = balanced_data['REASONFORCHANGE_trad'].apply(clean_stopwords)

    # print(balanced_data['DESCRIPTION_trad'][0:3])
    # Preprocesamiento para features
    print("Balanced data target count bf pf")
    print(balanced_data[target_col].value_counts())

    X, y, categorical_cols, text_cols, og_indices, txt_shape = prepare_features(balanced_data, original_data, embedder = 'w2v', target_col = target_col)
    
    class_names = ['False', 'True']
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 3. Training and Testing--------------------##
    sys.stderr.write(" -------------------- 3. Training  ----------------------------------\n")
    att = True
    metrics, model, test_loader, y_test = single_training_loop(X, y, args, balanced_data, original_data, nepoch, att, txt_shape)
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 4. Save metrics and predictions --------------------##
    sys.stderr.write(" -------------------- 4. Save metrics and predictions --------------------\n")
    model_path = evaluate_and_save_best_model_outputs(model, test_loader, y_test, balanced_data, original_data, class_names, target_col, name, att, txt_shape)

    csv_path = save_single_metrics(metrics, class_names, name)

    # save code
    code_path = '/fhome/amir/TFG/code/classificators/DL/DataQuality_classif.py'
    wandb.save(code_path)

    sys.stderr.write('----------------------------------------------------------------------\n')
    
    ## ------------- 5. Finalize WandB --------------------##
    sys.stderr.write(" -------------------- 5. Finalize WandB --------------------\n")
    wandb.save(os.path.join(os.path.dirname(__file__), 'DataQuality_classif.py'))
    wandb.finish()
    sys.stderr.write(f"\nBest model saved to {model_path}\n")
    sys.stderr.write(f"Metrics saved to {csv_path}\n")
    if args.save_fails:
        sys.stderr.write(f"Failed predictions saved to failed_predictions/ directory\n")
    sys.stderr.write("All done!\n")
    sys.stderr.write('----------------------------------------------------------------------\n')

if __name__ == "__main__":
    main()
