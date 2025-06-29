import argparse
import os
import wandb  # Para el tracking de experimentos

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fails', type=bool, default=False, 
                       help='Guardar registros con predicciones fallidas')
    return parser.parse_args()

def initialize_wandb(args, name_model,nepoch, extra):
    wandb.init(
        project="TFG",
        entity="andreu-mir-uab",
        name=name_model,
        config={
            "framework": "PyTorch",
            "model": "MLP2",
            "LOSS": "BCEweighted",
            "activation": "ReLU",
            "optimizer": "Adam",
            "max_epochs": nepoch,
            "patience": 10,
            "dataset": "CH_Total2",
            "balance_method": "RandomUnderSampling",
            "extras": extra,
            "save_fails": args.save_fails
        }
    )

def setup_preprocessing():
    os.makedirs('models/classifiers', exist_ok=True)
    os.makedirs('failed_predictions', exist_ok=True)
