# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import load_and_balance_data, load_and_balance_data_autorcross
# 3. PREPROCESS DATA
from functions.preprocess_data import prepare_features, install_nltk_resources, clean_stopwords

# 4. TRAIN/EVALUATION
from functions.training import single_training_loop, evaluate_and_save_best_model_outputs, save_single_metrics, authorcross_training_loop

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
    name = 'MLP2_CROSS_HUMANRPA2HUMAN_AE1BERT'  # Puedes cambiar esto según el cruce
    args = parse_arguments()
    nepoch = 200
    extra = 0
    initialize_wandb(args, name, nepoch, extra)

    ## ------------- 2. Load and preprocess data --------------------##
    sys.stderr.write(" -------------------- 2. Load and preprocess data --------------------\n")
    X_rpa, y_rpa, X_human, y_human, target_col = load_and_balance_data_autorcross(extra=extra)
    setup_preprocessing()

    embedder = 'AE1BERT' if 'AE1BERT' in name else 'BERT' if 'BERT' in name else 'w2v'
    
    # Preprocesamiento para features
    rpa_df = X_rpa.copy()
    rpa_df[target_col] = y_rpa
    human_df = X_human.copy()
    human_df[target_col] = y_human

    install_nltk_resources()
    # clean stopwords
    rpa_df['DESCRIPTION_trad'] = rpa_df['DESCRIPTION_trad'].apply(clean_stopwords)
    rpa_df['ITNDESCIMPACT_trad'] = rpa_df['ITNDESCIMPACT_trad'].apply(clean_stopwords)
    rpa_df['REASONFORCHANGE_trad'] = rpa_df['REASONFORCHANGE_trad'].apply(clean_stopwords)
    human_df['DESCRIPTION_trad'] = human_df['DESCRIPTION_trad'].apply(clean_stopwords)
    human_df['ITNDESCIMPACT_trad'] = human_df['ITNDESCIMPACT_trad'].apply(clean_stopwords)
    human_df['REASONFORCHANGE_trad'] = human_df['REASONFORCHANGE_trad'].apply(clean_stopwords)
    X_rpa_proc, y_rpa_proc, _, _, _, txt_shape = prepare_features(rpa_df, rpa_df, embedder=embedder)
    X_human_proc, y_human_proc, _, _, _, _ = prepare_features(human_df, human_df, embedder=embedder)

    sys.stderr.write(" -------------------- 3. Training (Cross Author) ---------------------\n")
    att = 'att' in name
    metrics, model, test_loader, y_test = authorcross_training_loop(
        X_rpa_proc, y_rpa_proc, X_human_proc, y_human_proc,
        args, nepoch, att=att, txt_shape=txt_shape, datatype_test = '2human'
    )

    ## ------------- 4. Save metrics and predictions --------------------##
    sys.stderr.write(" -------------------- 4. Save metrics and predictions --------------------\n")
    model_path = evaluate_and_save_best_model_outputs(model, test_loader, y_test, human_df, rpa_df, ['False', 'True'], target_col, name, att, txt_shape)
    csv_path = save_single_metrics(metrics, ['False', 'True'], name)
    wandb.save('/fhome/amir/TFG/code/classificators/DL/code/MLP2_main_authorcross.py')

    ## ------------- 5. Finalize WandB --------------------##
    sys.stderr.write(" -------------------- 5. Finalize WandB --------------------\n")
    wandb.finish()
    sys.stderr.write(f"\nBest model saved to {model_path}\n")
    sys.stderr.write(f"Metrics saved to {csv_path}\n")
    if args.save_fails:
        sys.stderr.write(f"Failed predictions saved to failed_predictions/ directory\n")
    sys.stderr.write("All done!\n")

if __name__ == "__main__":
    main()