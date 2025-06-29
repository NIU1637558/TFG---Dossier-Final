# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import load_and_balance_data

# 3. PREPROCESS DATA
from functions.preprocess_data import prepare_features

# 4. TRAIN/EVALUATION
from functions.training import single_training_loop, evaluate_and_save_best_model_outputs, save_single_metrics

# 5. MODELS
from model_arquitectres.models import MLP2  # Tu modelo de red neuronal personalizado

# others
import pandas as pd 
import numpy as np
import sys
import wandb


def main():
    ## ------------- 1. Initialize WandB --------------------##
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'MLP2_HUMAN_w2v'
    args = parse_arguments()
    nepoch = 200
    extra = 0
    initialize_wandb(args, name, nepoch, extra)

    ## ------------- 2. Load and preprocess data --------------------##
    sys.stderr.write(" -------------------- 2. Load and preprocess data --------------------\n")
    original_data, balanced_data, target_col = load_and_balance_data(extra, data_type = 'human')
    setup_preprocessing()

    # # sample balanced data
    # balanced_data = balanced_data.sample(n=1000, random_state=42)
    if 'w2v' in name:
        embedder = 'w2v'
    elif 'BERT' in name:
        embedder = 'BERT'
    else:
        embedder = 'AE'
    X, y, categorical_cols, text_cols, og_indices = prepare_features(balanced_data, original_data, embedder = 'w2v')
    
    class_names = ['False', 'True']
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 3. Training and Testing--------------------##
    sys.stderr.write(" -------------------- 3. Training  ----------------------------------\n")
    metrics, model, test_loader, y_test = single_training_loop(X, y, args, balanced_data, original_data, nepoch)
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 4. Save metrics and predictions --------------------##
    sys.stderr.write(" -------------------- 4. Save metrics and predictions --------------------\n")
    model_path = evaluate_and_save_best_model_outputs(model, test_loader, y_test, balanced_data, original_data, class_names, target_col, name)

    csv_path = save_single_metrics(metrics, class_names, name)

    # save code
    code_path = '/fhome/amir/TFG/code/classificators/DL/MLP_basic_H_w2v.py'
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
