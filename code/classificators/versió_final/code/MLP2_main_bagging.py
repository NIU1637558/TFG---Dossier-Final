# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import load_and_balance_data

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

def main():
    ## ------------- 1. Initialize WandB --------------------##
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'MLP2_HUMAN_DistilBERT_att_BAGGING'  # Cambiar nombre para reflejar que es bagging
    args = parse_arguments()
    nepoch = 150
    extra = 0
    initialize_wandb(args, name, nepoch, extra)

    ## ------------- 2. Load and preprocess data --------------------##
    sys.stderr.write(" -------------------- 2. Load and preprocess data --------------------\n")
    if 'RPA' in name:
        data_type = 'RPA'
    else:
        data_type = 'human'
    original_data, balanced_data, target_col = load_and_balance_data(extra, data_type)
    setup_preprocessing()
    embedder = None

    # 1. Word2Vec
    if 'w2v' in name:
        embedder = 'w2v'

    # 2. Doc2Vec
    if 'd2v' in name:
        embedder = 'd2v'

    # 3. BERT (MiniLM)
    if 'BERT' in name:
        embedder = 'BERT'

    # 4. Word2Vec + AE1
    if 'AE1w2v' in name:
        embedder = 'AE1w2v'

    # 5. BERT (MiniLM) + AE1
    if 'AE1BERT' in name:
        embedder = 'AE1BERT'

    # 6. DistilBERT + AE1
    if 'AE1DistlBERT' in name:
        embedder = 'AE1DistliBERT'
    
    # 7. BERT (MiniLM) finetuning mini + AE1
    if "AE1BERTtuned" in name:
        embedder = 'AE1BERTtuned'

    # 8. DistilBERT
    if "DistilBERT" in name:
        embedder = 'distBERT'

    # 9. W2V + RNN
    if "w2vRNN" in name:
        embedder = 'w2vRNN'

    if embedder == None:
        raise ValueError("El embedder especificat no es vàlid")

    install_nltk_resources()
    # # clean stopwords
    balanced_data['DESCRIPTION_trad'] = balanced_data['DESCRIPTION_trad'].apply(clean_stopwords)
    balanced_data['ITNDESCIMPACT_trad'] = balanced_data['ITNDESCIMPACT_trad'].apply(clean_stopwords)
    balanced_data['REASONFORCHANGE_trad'] = balanced_data['REASONFORCHANGE_trad'].apply(clean_stopwords)

    X, y, categorical_cols, text_cols, og_indices, txt_shape = prepare_features(balanced_data, original_data, embedder = embedder)
    
    class_names = ['False', 'True']
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 3. Training with Bagging --------------------##
    sys.stderr.write(" -------------------- 3. Training with Bagging -----------------------\n")
    if 'att' in name:
        att = True
        print('Attention Activated...')
    
    # Parámetros del bagging
    n_estimators = 20  # Número de modelos en el ensamblaje
    sample_ratio = 1  # Porcentaje de muestras para cada bootstrap
    
    # Entrenamiento con bagging
    metrics, ensemble, test_loader, y_test = bagging_training_loop(
        X, y, args, balanced_data, original_data, nepoch, 
        n_estimators=n_estimators, sample_ratio=sample_ratio, 
        att=att, txt_shape=txt_shape, hybrid = False
    )
    
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 4. Save metrics and predictions --------------------##
    sys.stderr.write(" -------------------- 4. Save metrics and predictions --------------------\n")
    model_path = evaluate_and_save_best_model_outputs(
        ensemble, test_loader, y_test, balanced_data, 
        original_data, class_names, target_col, name, False, txt_shape
    )

    csv_path = save_single_metrics(metrics, class_names, name)

    # save code
    code_path = '/fhome/amir/TFG/code/classificators/DL/MLP2_main_bagging.py'
    wandb.save(code_path)

    sys.stderr.write('----------------------------------------------------------------------\n')
    
    ## ------------- 5. Finalize WandB --------------------##
    sys.stderr.write(" -------------------- 5. Finalize WandB --------------------\n")
    wandb.save(os.path.join(os.path.dirname(__file__), 'MLP2_main_bagging.py'))
    wandb.finish()
    sys.stderr.write(f"\nEnsemble model saved to {model_path}\n")
    sys.stderr.write(f"Metrics saved to {csv_path}\n")
    if args.save_fails:
        sys.stderr.write(f"Failed predictions saved to failed_predictions/ directory\n")
    sys.stderr.write("All done!\n")
    sys.stderr.write('----------------------------------------------------------------------\n')

if __name__ == "__main__":
    main()