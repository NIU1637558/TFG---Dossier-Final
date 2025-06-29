# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import load_and_balance_data

# 3. PREPROCESS DATA
from functions.preprocess_data import prepare_features

# 4. TRAIN/EVALUATION
from functions.training import single_training_loop, evaluate_and_save_best_model_outputs, save_single_metrics, hybrid_training_loop_multiinput, evaluate_and_save_best_model_outputs_multiinput

# 5. MODELS
from model_arquitectres.models import MLP2, HybridAttentionMLP, HybridNeuralStacking

# others
import pandas as pd 
import numpy as np
import sys
import wandb
import os
import torch

# main.py


def load_models(model_paths, macroX, macro_txtshape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_loaded = {}
    
    for key, paths in model_paths.items():
        input_size = macroX[key].shape[1] - macro_txtshape[key]
        text_shape = macro_txtshape[key]

        loaded_models = []
        for path in paths:
            # Cargar los datos guardados
            checkpoint = torch.load(path, map_location=device)

            # Instanciar el modelo con la misma arquitectura
            model = HybridAttentionMLP(input_size, text_shape, dropout=0.8, hybrid=False)
            
            # Cargar solo los pesos
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            loaded_models.append(model)

        models_loaded[key] = loaded_models

    return models_loaded


def main():
    ## ------------- 1. Initialize WandB --------------------##
    sys.stderr.write(" -------------------- 1. Initialize WandB ---------------------------\n")
    name = 'Hybrid_mlp1_HUMAN'
    args = parse_arguments()
    nepoch = 200
    n_models = 2 # num models per type, max 8
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

    # # # sample balanced data
    # balanced_data = balanced_data.sample(n=1000, random_state=42)
    
    class_names = ['False', 'True']

    # get embeddings for each model
    macroX = {}
    macro_txtshape = {}
    embedders = ['w2v', 'BERT', 'AE1BERT']
    for embedder in embedders:
        print(f'Using {embedder} embedding...')
        macroX[embedder], y, categorical_cols, text_cols, og_indices, macro_txtshape[embedder] = prepare_features(balanced_data, original_data, embedder=embedder)

    sys.stderr.write('---------------------------------------------------------------------\n')

    ## ------------- 3. Create Hibrid Model --------------------##
    sys.stderr.write(" -------------------- 3. Create Hibrid Model --------------------\n")

    models = {
        "w2v": [f"/fhome/amir/TFG/models/classifiers/MLP_HUMAN_w2v_{i+1}_att.pth" for i in range(n_models)],
        # "d2v": [f"/fhome/amir/TFG/models/classifiers/MLP_HUMAN_d2v_{i+1}_att.pth" for i in range(n_models)],
        "BERT": [f"/fhome/amir/TFG/models/classifiers/MLP_HUMAN_MiniLM_{i+1}_att.pth" for i in range(n_models)],
        "AE1BERT": [f"/fhome/amir/TFG/models/classifiers/MLP_HUMAN_AE1MiniLM_{i+1}_att.pth" for i in range(n_models)],
    #     "DistilBERT": [f"/fhome/amir/TFG/models/classifiers/MLP2_HUMAN_DistilBERT_{i}.pth" for i in enumerate(range(n_models))],
    #     "AE1DistilBERT": [f"/fhome/amir/TFG/models/classifiers/MLP2_HUMAN_AE1DistilBERT_{i}.pth" for i in enumerate(range(n_models))],
    #     "MiniLMtuned": [f"/fhome/amir/TFG/models/classifiers/MLP2_HUMAN_MiniLMtuned_{i}.pth" for i in enumerate(range(n_models))],
    #     "AE1MiniLMtuned": [f"/fhome/amir/TFG/models/classifiers/MLP2_HUMAN_AE1MiniLMtuned_{i}.pth" for i in enumerate(range(n_models))],
    }

    models = {
        "w2v": ["/fhome/amir/TFG/models/hybrid_neurons_pretrained/MLP2_HUMAN_w2v_1_att.pth",
                "/fhome/amir/TFG/models/hybrid_neurons_pretrained/MLP2_HUMAN_w2v_2_att.pth"],
        "BERT": ["/fhome/amir/TFG/models/hybrid_neurons_pretrained/MLP2_HUMAN_BERT_1_att.pth",
                "/fhome/amir/TFG/models/hybrid_neurons_pretrained/MLP2_HUMAN_BERT_2_att.pth"],
        "AE1BERT": ["/fhome/amir/TFG/models/hybrid_neurons_pretrained/MLP2_HUMAN_AE1BERT_1_att.pth",
                    "/fhome/amir/TFG/models/hybrid_neurons_pretrained/MLP2_HUMAN_AE1BERT_2_att.pth"],
            }
    
    models_loaded = load_models(models, macroX, macro_txtshape)

    ## ------------- 4. Training and Testing--------------------##
    sys.stderr.write(" -------------------- 3. Training  ----------------------------------\n")
    if 'att' in name:
        att = True
        print('Attention Activated...')
    metrics, model, test_loader, y_test = hybrid_training_loop_multiinput(macroX, y, models_loaded, macro_txtshape, nepoch=nepoch)
    sys.stderr.write('---------------------------------------------------------------------\n')
    
    ## ------------- 5. Save metrics and predictions --------------------##
    sys.stderr.write(" -------------------- 4. Save metrics and predictions --------------------\n")
    model_path = evaluate_and_save_best_model_outputs_multiinput(
        model, test_loader, y_test,
        balanced_data, original_data,
        class_names, target_col, name_model=name
    )


    csv_path = save_single_metrics(metrics, class_names, name)

    # save code
    code_path = '/fhome/amir/TFG/code/classificators/DL/code/hibrid_main.py'
    wandb.save(code_path)

    sys.stderr.write('----------------------------------------------------------------------\n')
    
    ## ------------- 6. Finalize WandB --------------------##
    sys.stderr.write(" -------------------- 5. Finalize WandB --------------------\n")
    wandb.save(os.path.join(os.path.dirname(__file__), 'hibrid_main.py'))
    wandb.finish()
    sys.stderr.write(f"\nBest model saved to {model_path}\n")
    sys.stderr.write(f"Metrics saved to {csv_path}\n")
    if args.save_fails:
        sys.stderr.write(f"Failed predictions saved to failed_predictions/ directory\n")
    sys.stderr.write("All done!\n")
    sys.stderr.write('----------------------------------------------------------------------\n')

if __name__ == "__main__":
    main()