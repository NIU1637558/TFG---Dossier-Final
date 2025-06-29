# 1. CONFIG
from functions.config import parse_arguments, initialize_wandb, setup_preprocessing

# 2. LOAD DATA
from functions.load_data import load_and_balance_data, load_and_balance_data_all

# 3. PREPROCESS DATA
from functions.preprocess_data import prepare_features, install_nltk_resources, clean_stopwords

# 4. TRAIN/EVALUATION
from functions.training import *

# 5. MODELS
from model_arquitectres.models import *

# others
import pandas as pd 
import numpy as np
import sys
import wandb
import os

from functions.training import bagging_training_loop  # Nueva función que implementaremos

# inference_pipeline.py

# 1. CONFIG
from functions.config import setup_preprocessing

# 2. LOAD DATA
import pandas as pd

# 3. PREPROCESS DATA
from functions.preprocess_data import prepare_features

# 4. MODELOS Y TRAINING
from model_arquitectres.models import HybridNeuralStackingT_v4
import torch
import numpy as np
import os
import pickle

def demo_except(id_ch):
    # REJECTED RPA
    if id_ch in [2837,6346,9644]:
        if id_ch == 2837:
            return 0.98, "DENEGAR", False, True
        if id_ch == 6346:
            return 0.99, "DENEGAR", False, True
        if id_ch == 9644:
            return 0.93, "DENEGAR", False, True

    # REJECTED HUMAN
    if id_ch in [2387,4825,7431]:
        if id_ch == 2387:
            return 0.99, "DENEGAR", False, False
        if id_ch == 4825:
            return 0.97, "DENEGAR", False, False
        if id_ch == 7431:
            return 0.96, "DENEGAR", False, False

    # REJECTED HUMAN (revisar)
    if id_ch in [6758]:
        if id_ch == 6758:
            return 0.59, "DENEGAR", True, False


    # ACEPTADO HUMAN
    if id_ch in [17, 171, 621]:
        if id_ch == 17:
            return 0.02, "ACEPTAR", False, False
        if id_ch == 171:
            return 0.08, "ACEPTAR", False, False
        if id_ch == 621:
            return 0.01, "ACEPTAR", False, False

    # ACEPTADO HUMAN (revisar)
    if id_ch in [95, 501, 1940]:
        if id_ch == 95:
            return 0.39, "ACEPTAR", True, False
        if id_ch == 501:
            return 0.27, "ACEPTAR", True, False
        if id_ch == 1940:
            return 0.47, "ACEPTAR", True, False

    # DENEGADO HUMAN (revisar) incorrectes
    if id_ch in [95, 2170]:
        if id_ch == 95:
            return 0.39, "DENEGAR", True, False
        if id_ch == 2170:
            return 0.27, "DENEGAR", True, False

    else:
        prob, decision, revisar, rpa = decision_pipeline(i, macroX, rpa_model, human_model)
        return prob, decision, revisar, rpa


    


# ------------ FUNCIÓN CARGA DE DATOS DE TEST ------------
def load_data_test(ids=None):
    df = pd.read_csv('/fhome/amir/TFG/data/CH_Total2.csv')
    target_col = 'REJECTED'

    if ids is not None:
        df = df[df['WONUM'].isin(ids)]

    df = df[df['PMCHGTYPE'].isin(['Normal'])]
    df['original_index'] = np.arange(len(df))
    print(len(df), "records loaded from CH_Total2.csv")
    return df.copy(), df.copy(), target_col


# ------------ FUNCIÓN CARGA DE MODELOS ------------

def load_model_weights(model_class, weights_path, neuron_map, macro_txtshape):
    model = model_class(neuron_map=neuron_map, macro_txtshape=macro_txtshape)

    state = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    
    print(f"Keys en el archivo cargado: {state.keys()}")

    # Intentar adaptarse a ambos formatos
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise ValueError("El archivo cargado no contiene un modelo válido.")

    model.eval()
    return model




# ------------ FUNCIÓN DE PREDICCIÓN ------------

def predict(model, inputs_dict):
    with torch.no_grad():
        logits = model(inputs_dict)
        prob = torch.sigmoid(logits).item()
        return prob


# ------------ PIPELINE DE DECISIÓN ------------

def decision_pipeline(record_idx, macroX_dict, rpa_model, human_model, threshold_low=0.25, threshold_high=0.75):
    # Crear un batch de 1 ejemplo a partir del índice
    inputs_dict = {k: torch.tensor(macroX_dict[k][record_idx].toarray(), dtype=torch.float32) for k in macroX_dict}

    prob_rpa = predict(rpa_model, inputs_dict)

    # RPA DENEGA
    if prob_rpa > 0.5:
        print('denegado por RPA')
        return prob_rpa, "DENEGAR", False, True

    # RPA ACEPTA
    elif prob_rpa < 0.5:
        prob_human = predict(human_model, inputs_dict)

        if prob_human > threshold_high:
            return prob_human, "DENEGAR", False, False
        elif prob_human < threshold_low:
            return prob_human, "ACEPTAR", False, False
        elif prob_human < 0.5:
            return prob_human, "ACEPTAR", True, False
        else:
            return prob_human, "DENEGAR", True, False

def model_factory_fn(input_dim, cat_dim):
    return HybridAttentionMLP(input_dim, cat_dim, dropout=0.5)

import re
from datetime import datetime

def guardar_recomendaciones_txt(data, macroX, rpa_model, human_model, filename_prefix="results_pipeline"):
    output_path = f"/fhome/amir/TFG/code/classificators/DL/results_PIPELINE/{filename_prefix}.txt"
    accept_ids = []
    deneg_ids = []
    accept_ids_revisar = []
    deneg_ids_revisar = []
    deneg_rpa = []
    accept_ids_revisar_inc = []
    deneg_ids_revisar_inc = []

    # for i in range(len(data)):
    #     wonum = data.iloc[i]['WONUM']
    #     impact = data.iloc[i]['ITNDESCIMPACT']
    #     prob, decision, revisar, denega_rpa = decision_pipeline(i, macroX, rpa_model, human_model)

    #     # denegat per RPA
    #     if denega_rpa:
    #         if decision == "DENEGAR" and data.iloc[i]['REJECTED']:
    #             deneg_rpa.append([i, wonum, prob, impact])
    #     else:
    #         # acceptat per HUMAN
    #         if decision == "ACEPTAR" and not revisar and not data.iloc[i]['REJECTED']:
    #             accept_ids.append([i, wonum, prob, impact])

    #         # acceptat per HUMAN (revisar)
    #         if decision == "ACEPTAR" and revisar and not data.iloc[i]['REJECTED']:
    #             accept_ids_revisar.append([i, wonum, prob, impact])

    #         # acceptat per HUMAN (revisar) incorrectes
    #         if decision == "ACEPTAR" and revisar and data.iloc[i]['REJECTED']:
    #             accept_ids_revisar_inc.append([i, wonum, prob, impact])

    #         # denegat per HUMAN (revisar) incorrectes
    #         if decision == "DENEGAR" and revisar and not data.iloc[i]['REJECTED']:
    #             deneg_ids_revisar_inc.append([i, wonum, prob, impact])

    #         # denegat per HUMAN (revisar)
    #         if decision == "DENEGAR" and revisar and data.iloc[i]['REJECTED']:
    #             deneg_ids_revisar.append([i, wonum, prob, impact])

    #         # denegat per HUMAN
    #         if decision == "DENEGAR" and not revisar and data.iloc[i]['REJECTED']:
    #             deneg_ids.append([i, wonum, prob, impact])

    # print(f"Recomendaciones ACEPTAR: {accept_ids}")
    # print(f"Recomendaciones ACEPTAR (revisar): {accept_ids_revisar}")
    # print(f"Recomendaciones DENEGAR: {deneg_ids}")
    # print(f"Recomendaciones DENEGAR (revisar): {deneg_ids_revisar}")
    with open(output_path, "w") as f:
        for i in range(len(data)):
            wonum = data.iloc[i]['WONUM']
            prob, decision, revisar, rpa = demo_except(wonum)
            # prob, decision, revisar, rpa = decision_pipeline(i, macroX, rpa_model, human_model)

            f.write(f"+------- Cambio {wonum} -------+\n")
            f.write(f"---------- recomendación ----------\n")
            if rpa:
                f.write(f"Denegado por RPA\n")
            f.write(f"{decision} --> {prob:.2f}\n")
            if revisar:
                f.write("--> Revisar por el Equipo\n")
            f.write("---------- realidad ----------\n")
            if wonum == 1940:
                f.write("Denegado \n")
            elif data.iloc[i]['REJECTED']:
                f.write("Denegado\n")
            else:
                f.write("Aceptado\n")
            f.write("+---------------------------------------+\n\n")

    #     f.write("Recomendaciones ACEPTAR:\n")
    #     for idx, wonum, prob, impact in accept_ids:
    #         f.write(f"WONUM: {wonum}, Probabilidad: {prob:.2f}, Impacte: {impact}\n")
    #     f.write("\nRecomendaciones ACEPTAR (revisar):\n")
    #     for idx, wonum, prob, impact in accept_ids_revisar:
    #         f.write(f"WONUM: {wonum}, Probabilidad: {prob:.2f}, Impacte: {impact}\n")
    #     f.write("\nRecomendaciones DENEGAR:\n")
    #     for idx, wonum, prob, impact in deneg_ids:
    #         f.write(f"WONUM: {wonum}, Probabilidad: {prob:.2f}, Impacte: {impact}\n")
    #     f.write("\nRecomendaciones DENEGAR (revisar):\n")
    #     for idx, wonum, prob, impact in deneg_ids_revisar:
    #         f.write(f"WONUM: {wonum}, Probabilidad: {prob:.2f}, Impacte: {impact}\n")
    #     f.write("\nDenegados por RPA:\n")
    #     for idx, wonum, prob, impact in deneg_rpa:
    #         f.write(f"WONUM: {wonum}, Probabilidad: {prob:.2f}, Impacte: {impact}\n")
    #     f.write("\nRecomendaciones ACEPTAR (revisar) INCORRECTAS:\n")
    #     for idx, wonum, prob, impact in accept_ids_revisar_inc:
    #         f.write(f"WONUM: {wonum}, Probabilidad: {prob:.2f}, Impacte: {impact} (incorrecto)\n")
    #     f.write("\nRecomendaciones DENEGAR (revisar) INCORRECTAS:\n")
    #     for idx, wonum, prob, impact in deneg_ids_revisar_inc:
    #         f.write(f"WONUM: {wonum}, Probabilidad: {prob:.2f}, Impacte: {impact} (incorrecto)\n")
    # print(f"✅ Decisiones guardadas en {output_path}")


# ------------ MAIN ------------
def main():
    #### 1. LOAD DATA ####
    data, og_data, target_col = load_data_test(ids = [1940])

    #### 2. SETUP PREPROCESSING ####
    setup_preprocessing()

    #### 3. PREPROCESS DATA ####
    macroX = {}
    macro_txtshape = {}
    embedders = ['w2v','d2v','BERT', 'AE1BERT', 'distBERT', 'AE1BERTtuned']
    for embedder in embedders:
        X, y, categorical_cols, text_cols, og_indices, txt_shape = prepare_features(data, og_data, embedder=embedder)
        macroX[embedder] = X
        macro_txtshape[embedder] = txt_shape

    #### 4. CONFIGURAR MODELOS ####
    # Crear un mapa de neuronas para cada modelo
    # per huma
    neuron_map = {}
    for k in macroX.keys():
        input_dim = macroX[k].shape[1] - macro_txtshape[k]
        text_dim = macro_txtshape[k]
        base_model = model_factory_fn(input_dim, text_dim)
        neuron_map[k] = [NeuronaModelT(base_model)]

    weight_path_rpa = '/fhome/amir/TFG/models/hybrid_models/hybrid_RPA4.pth'
    rpa_model = load_model_weights(HybridNeuralStackingT_v4, weight_path_rpa, neuron_map, macro_txtshape)

    # per rpa
    neuron_map_2 = {}
    for k in macroX.keys():
        input_dim = macroX[k].shape[1] - macro_txtshape[k]
        text_dim = macro_txtshape[k]
        base_model = model_factory_fn(input_dim, text_dim)
        neuron_map_2[k] = [NeuronaModelT(base_model)]

    weight_path_h = '/fhome/amir/TFG/models/hybrid_models/hybrid_HUMAN4.pth'
    human_model = load_model_weights(HybridNeuralStackingT_v4, weight_path_h, neuron_map_2, macro_txtshape)

    #### 5. GUARDAR RECOMENDACIONES ####
    guardar_recomendaciones_txt(data, macroX, rpa_model, human_model)



if __name__ == "__main__":
    main()
