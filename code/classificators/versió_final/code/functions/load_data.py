import pandas as pd
from imblearn.under_sampling import RandomUnderSampler  # Para el balanceo de datos
import numpy as np
from torch.utils.data import Dataset
import torch

def filter_out_RPA_rejected(dataframe):
    """
    Elimina filas donde REJECTED es RPA.
    """
    # filter ALL RPA
    mask = ~((dataframe['author'].str.upper() == 'RPA'))

    # # filter only REJECTED RPA
    # mask = ~((dataframe['REJECTED'] == True) & (dataframe['author'] == 'RPA'))

    return dataframe[mask].reset_index(drop=True)

def filter_out_HUMAN_rejected(dataframe):
    """
    Elimina filas donde REJECTED es HUMAN.
    """
    # filter ALL HUMAN
    mask = ~((dataframe['author'].str.upper() == 'HUMAN'))

    # # filter only REJECTED HUMAN
    # mask = ~((dataframe['REJECTED'] == True) & (dataframe['author'] == 'HUMAN'))

    return dataframe[mask].reset_index(drop=True)

def load_and_balance_data_all(extra,dq = False):
    print("Loading and preprocessing original data (ALL AUTHORS)...")
    original_data0 = pd.read_csv('/fhome/amir/TFG/data/CH_Total2.csv')
    target_col = 'REJECTED'

    print(original_data0['author'].value_counts())
    print(original_data0.columns)

    # per DATAQUALITY
    if dq:
        # original_data0['PROD_CAIXA'] = original_data0['ITNESCENARIO'] == 'CAT_0945bfc0'
        # original_data0 = original_data0.drop(columns=['ITNESCENARIO'])
        # target_col = 'PROD_CAIXA'

        original_data0['NULL_IMPACT'] = original_data0['IMPACT_CAT'] == 'Null'
        original_data0 = original_data0.drop(columns=['IMPACT_CAT'])
        target_col = 'NULL_IMPACT'

    # Filtrar tipos de cambio relevantes
    original_data1 = original_data0[original_data0['PMCHGTYPE'].isin(['Normal'])]
    print(f"Total Data with filter1: {original_data1.shape[0]}")
    print(original_data1['author'].value_counts())

    original_data = original_data1.copy()  # No filtramos HUMAN ni RPA
    print(f"Total Data without author filtering: {original_data.shape[0]}")

    # Guardar el índice original como columna antes del muestreo
    original_data = original_data.copy()
    original_data['original_index'] = original_data.index

    # Sample extra ejemplos de clase 0
    if extra > 0:
        rejected_false = original_data[original_data[target_col] == 0].sample(n=extra, random_state=42)
        original_data = original_data.drop(rejected_false.index)

    # Separar features y target
    X_temp = original_data.drop(columns=[target_col])
    y_temp = original_data[target_col]

    # Undersampling
    undersampler = RandomUnderSampler(sampling_strategy = {0: 5000, 1: 5000}) 
    # undersampler = RandomUnderSampler()
    X_balanced, y_balanced = undersampler.fit_resample(X_temp, y_temp)

    print('Total Data after undersampling:', X_balanced.shape[0])

    # Recuperar índice original desde la columna guardada
    balanced_data = pd.concat([X_balanced, y_balanced], axis=1)

    # Si hay ejemplos extra, los añadimos con su índice original
    if extra > 0:
        rejected_false = rejected_false.copy()
        rejected_false['original_index'] = rejected_false.index
        balanced_data = pd.concat([balanced_data, rejected_false], axis=0)

    balanced_data.reset_index(drop=True, inplace=True)

    print(f"Total Final Data: {balanced_data.shape[0]}")
    # show how many registerts per target
    print("Target column:", target_col)
    print(balanced_data[target_col].value_counts())

    # # sample 100 rows
    # balanced_data = balanced_data.sample(n=min(10, balanced_data.shape[0]), random_state=42)
    # original_data = original_data.sample(n=min(10, original_data.shape[0]), random_state=42)

    return original_data, balanced_data, target_col


def load_and_balance_data(extra, data_type, dq = False):
    print("Loading and preprocessing original data...")
    original_data0 = pd.read_csv('/fhome/amir/TFG/data/CH_Total2.csv')
    target_col = 'REJECTED'

    print(original_data0['author'].value_counts())

    # per DATAQUALITY
    if dq:
        # original_data0['PROD_CAIXA'] = original_data0['ITNESCENARIO'] == 'CAT_0945bfc0'
        # original_data0 = original_data0.drop(columns=['ITNESCENARIO'])
        # target_col = 'PROD_CAIXA'

        original_data0['NULL_IMPACT'] = original_data0['IMPACT_CAT'] == 'Null'
        original_data0 = original_data0.drop(columns=['IMPACT_CAT'])
        target_col = 'NULL_IMPACT'

    # Filtrar tipos de cambio relevantes
    original_data1 = original_data0[original_data0['PMCHGTYPE'].isin(['Normal'
    ])]
    # original_data1 = original_data0.copy()

    print(f"Total Data with filter1: {original_data1.shape[0]}")

    print(original_data1['author'].value_counts())

    # Filtrar HUMAN rejected
    if data_type == 'human':
        original_data = filter_out_RPA_rejected(original_data1)
        print('ogdata',original_data['author'].value_counts())
    else:
        original_data = filter_out_HUMAN_rejected(original_data1)
        print('ogdata',original_data['author'].value_counts())

    print(f"Total Data with filter2: {original_data.shape[0]}")

    # Guardar el índice original como columna antes del muestreo
    original_data = original_data.copy()
    original_data['original_index'] = original_data.index

    # Sample extra ejemplos de clase 0
    if extra > 0:
        rejected_false = original_data[original_data[target_col] == 0].sample(n=extra, random_state=42)
        original_data = original_data.drop(rejected_false.index)

    # Separar features y target
    X_temp = original_data.drop(columns=[target_col])
    y_temp = original_data[target_col]

    # Undersampling
    # undersampler = RandomUnderSampler(sampling_strategy = {0: 5000, 1: 5000}) 
    undersampler = RandomUnderSampler()
    X_balanced, y_balanced = undersampler.fit_resample(X_temp, y_temp)

    print('Total Data after undersampling:', X_balanced.shape[0])

    # Recuperar índice original desde la columna guardada
    balanced_data = pd.concat([X_balanced, y_balanced], axis=1)
    print(balanced_data.columns)

    # Si hay ejemplos extra, los añadimos con su índice original
    if extra > 0:
        rejected_false = rejected_false.copy()
        rejected_false['original_index'] = rejected_false.index
        balanced_data = pd.concat([balanced_data, rejected_false], axis=0)

    balanced_data.reset_index(drop=True, inplace=True)

    print(f"Total Data: {balanced_data.shape[0]}")

    # # sample 10 rows
    # balanced_data = balanced_data.sample(n=min(10, balanced_data.shape[0]), random_state=42)
    # original_data = original_data.sample(n=min(10, original_data.shape[0]), random_state=42)

    return original_data, balanced_data, target_col

def load_and_balance_data_autorcross(extra=0):
    print("Loading and preprocessing original data...")
    original_data0 = pd.read_csv('/fhome/amir/TFG/data/CH_Total2.csv')
    target_col = 'REJECTED'

    # Filtro por tipo de cambio
    original_data1 = original_data0[original_data0['PMCHGTYPE'].isin(['Normal'])]
    print(f"Total Data with filter1: {original_data1.shape[0]}")

    # Filtrar datos HUMAN
    human_data = filter_out_RPA_rejected(original_data1)
    print('HUMAN count:', human_data['author'].value_counts())
    human_data = human_data.copy()
    human_data['original_index'] = human_data.index

    # Filtrar datos RPA
    rpa_data = filter_out_HUMAN_rejected(original_data1)
    print('RPA count:', rpa_data['author'].value_counts())
    rpa_data = rpa_data.copy()
    rpa_data['original_index'] = rpa_data.index

    # Opcional: remover ejemplos extra de clase 0
    if extra > 0:
        rejected_false_rpa = rpa_data[rpa_data[target_col] == 0].sample(n=extra, random_state=42)
        rpa_data = rpa_data.drop(rejected_false_rpa.index)
        rejected_false_human = human_data[human_data[target_col] == 0].sample(n=extra, random_state=42)
        human_data = human_data.drop(rejected_false_human.index)
    else:
        rejected_false_rpa = None
        rejected_false_human = None

    # Separar X, y de RPA
    X_rpa_temp = rpa_data.drop(columns=[target_col])
    y_rpa_temp = rpa_data[target_col]
    undersampler_rpa = RandomUnderSampler()
    X_rpa, y_rpa = undersampler_rpa.fit_resample(X_rpa_temp, y_rpa_temp)
    balanced_rpa = pd.concat([X_rpa, y_rpa], axis=1)
    if extra > 0 and rejected_false_rpa is not None:
        rejected_false_rpa['original_index'] = rejected_false_rpa.index
        balanced_rpa = pd.concat([balanced_rpa, rejected_false_rpa], axis=0)
    balanced_rpa.reset_index(drop=True, inplace=True)
    y_rpa = balanced_rpa[target_col]
    X_rpa = balanced_rpa.drop(columns=[target_col])

    # Separar X, y de HUMAN
    X_human_temp = human_data.drop(columns=[target_col])
    y_human_temp = human_data[target_col]
    undersampler_human = RandomUnderSampler()
    X_human, y_human = undersampler_human.fit_resample(X_human_temp, y_human_temp)
    balanced_human = pd.concat([X_human, y_human], axis=1)
    if extra > 0 and rejected_false_human is not None:
        rejected_false_human['original_index'] = rejected_false_human.index
        balanced_human = pd.concat([balanced_human, rejected_false_human], axis=0)
    balanced_human.reset_index(drop=True, inplace=True)
    y_human = balanced_human[target_col]
    X_human = balanced_human.drop(columns=[target_col])

    print(f"Total RPA: {X_rpa.shape[0]}")
    print(f"Total HUMAN: {X_human.shape[0]}")

    return X_rpa, y_rpa, X_human, y_human, target_col
