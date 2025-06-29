# imports data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import re

# imports embedding + training
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
import torch
import pickle
from tqdm import tqdm

print("Imports done")

# 1. Load data
print("Loading data...")
dataOG = pd.read_csv('/fhome/amir/TFG/data/CH_Total_label2_trad.csv')

# # sample only 5000 registers
# dataOG = dataOG.sample(n=5000, random_state=42)

# 2. Get impact labels
# 1. Tots a Other
data = dataOG.copy()
data['IMPACT_CAT'] = 'Other'

# 2. Definim keywords
null_impact_keywords = [
    'nulo', 'nula', 'ninguno', 'ninguna', 'null', 'none', 'nada', 'cero', 'insignificante',  # Español/Inglés
    'nulo', 'nula', 'nenhum', 'nenhuma', 'nulo', 'nada', 'zero', 'insignificante'  # Portugués
]

low_impact_keywords = [
    'bajo', 'baja', 'leve', 'leves', 'mínimo', 'mínima', 'mínimos', 'mínimas',
    'mínimamente', 'menor', 'menores', 'low',  # Español/Inglés
    'baixo', 'baixa', 'leve', 'leves', 'mínimo', 'mínima', 'mínimos', 'mínimas',
    'mínimamente', 'menor', 'menores',  # Portugués
    'no debería', 'no deberia', 'no deberían', 'no deberian', 'no deberíamos', 'no deberiamos',  # Español II
    'não deveria', 'não deveriam', 'não deveríamos', 'não deveriamos'  # Portugués II
]

medium_impact_keywords = [
    'medio', 'media', 'moderado', 'moderada', 'intermedio', 'intermedia',
    'moderadamente', 'medium',  # Español/Inglés
    'médio', 'média', 'moderado', 'moderada', 'intermediário', 'intermediária',
    'moderadamente'  # Portugués
]

high_impact_keywords = [
    'alto', 'alta', 'grave', 'graves', 'severo', 'severa', 'crítico', 'crítica', 'high',  # Español/Inglés
    'alto', 'alta', 'grave', 'graves', 'severa', 'severa', 'crítico', 'crítica'  # Portugués
]

key_words_extra_null = [
    'no hay', 'no existe', 'no se', 'sin'  # Español
    'não há', 'não existe', 'não se', 'sem'  # Portugués
]

# Función para tokenizar el texto
def tokenize_text(text):
    if isinstance(text, str):
        # Tokenización simple: dividir por espacios y eliminar puntuación
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    return []

# Aplicar la función para tokenizar ITNDESCIMPACT
data['TOKENS'] = data['ITNDESCIMPACT'].apply(tokenize_text)

# Función para verificar si alguna palabra clave está en los tokens
def contains_keywords(tokens, keywords):
    return any(token in keywords for token in tokens)

# Convertir las listas de palabras clave a conjuntos para una búsqueda más eficiente
null_keywords_set = set(null_impact_keywords)
low_keywords_set = set(low_impact_keywords)
medium_keywords_set = set(medium_impact_keywords)
high_keywords_set = set(high_impact_keywords)
extra_null_keywords_set = set(key_words_extra_null)

# 3. Categorizar NULL basado en tokens
data.loc[
    (data['TOKENS'].apply(contains_keywords, keywords=null_keywords_set)) &
    (data['IMPACT_CAT'] == 'Other'),
    'IMPACT_CAT'
] = 'Null'

# 4. Categorizar LOW basado en tokens
data.loc[
    (data['TOKENS'].apply(contains_keywords, keywords=low_keywords_set)) &
    (data['IMPACT_CAT'] == 'Other'),
    'IMPACT_CAT'
] = 'Low'

# 5. Categorizar MEDIUM basado en tokens
data.loc[
    (data['TOKENS'].apply(contains_keywords, keywords=medium_keywords_set)) &
    (data['IMPACT_CAT'] == 'Other'),
    'IMPACT_CAT'
] = 'Medium'

# 6. Categorizar HIGH basado en tokens
data.loc[
    (data['TOKENS'].apply(contains_keywords, keywords=high_keywords_set)) &
    (data['IMPACT_CAT'] == 'Other'),
    'IMPACT_CAT'
] = 'High'

# 7. Categorizar NULL II basado en tokens (key_words_extra_null)
data.loc[
    (data['ITNDESCIMPACT'].str.lower().str.contains('|'.join(key_words_extra_null), case = False, na=False)) &
    (data['IMPACT_CAT'] == 'Other'),
    'IMPACT_CAT'
] = 'Null'

# save to CH_trad_impactCat.csv
data.to_csv('/fhome/amir/TFG/data/CH_trad_impactCat.csv', index=False)

# 3. EMbeddings

# posem nulls a str per label encoder
data.fillna("None", inplace=True)

# Configuración inicial
os.makedirs('models', exist_ok=True)

# Columnas categóricas
categorical_cols = [
    'WONUM', 'STATUS', 'ITNCHGCOMCANAL', 'ITNESCENARIO', 'ACTSTART', 'PMCHGTYPE',
    'OWNERGROUP', 'ORIGRECORDCLASS', 'ORIGRECORDID', 'PMCHGCONCERN', 'ITDCLOSURECODE',
    'PROBLEMCODE', 'ITDCHCREATEDBYGROUP', 'ACTFINISH', 'ENVENTANA'
]

# Verificar que las columnas categóricas sean de tipo 'object' o 'category'
for col in categorical_cols:
    if data[col].dtype != 'object' and data[col].dtype.name != 'category':
        data[col] = data[col].astype('object')

# Variable objetivo
target_col = 'IMPACT_CAT'

# Cargar el modelo de Sentence Transformers (mirar de canviar num embeddings)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Generar embeddings
print("Generando embeddings...")
description_embeddings = model.encode(data['DESCRIPTION'].tolist(), show_progress_bar=True, device=device)
itndescimpact_embeddings = model.encode(data['ITNDESCIMPACT'].tolist(), show_progress_bar=True, device=device)
reasonforchange_embeddings = model.encode(data['REASONFORCHANGE'].tolist(), show_progress_bar=True, device=device)  

# Convertir embeddings a DataFrames
description_embeddings_df = pd.DataFrame(description_embeddings, columns=[f'desc_embed_{i}' for i in range(description_embeddings.shape[1])])
itndescimpact_embeddings_df = pd.DataFrame(itndescimpact_embeddings, columns=[f'impact_embed_{i}' for i in range(itndescimpact_embeddings.shape[1])])
reasonforchange_embeddings_df = pd.DataFrame(reasonforchange_embeddings, columns=[f'reason_embed_{i}' for i in range(reasonforchange_embeddings.shape[1])])

# 4. LAbel Encoder
# Aplicar Label Encoding a las columnas categóricas
print("Aplicando Label Encoding a las columnas categóricas...")
label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoded_categorical = label_encoder.fit_transform(data[categorical_cols])

# Convertir todo a matrices dispersas
description_embeddings_sparse = csr_matrix(description_embeddings_df.astype(np.float32))
itndescimpact_embeddings_sparse = csr_matrix(itndescimpact_embeddings_df.astype(np.float32))
reasonforchange_embeddings_sparse = csr_matrix(reasonforchange_embeddings_df.astype(np.float32))
encoded_categorical_sparse = csr_matrix(encoded_categorical)

# 5. Model COmbinations
# Definir las combinaciones de características
combinations = {
    'modelo1': [encoded_categorical_sparse, description_embeddings_sparse, itndescimpact_embeddings_sparse, reasonforchange_embeddings_sparse],
    'modelo2': [encoded_categorical_sparse, description_embeddings_sparse, reasonforchange_embeddings_sparse],
    'modelo3': [encoded_categorical_sparse, description_embeddings_sparse, itndescimpact_embeddings_sparse],
    'modelo4': [encoded_categorical_sparse, itndescimpact_embeddings_sparse, reasonforchange_embeddings_sparse],
    'modelo5': [encoded_categorical_sparse, itndescimpact_embeddings_sparse],
    'modelo6': [encoded_categorical_sparse, description_embeddings_sparse],
    'modelo7': [encoded_categorical_sparse, reasonforchange_embeddings_sparse],
    'modelo8': [description_embeddings_sparse, itndescimpact_embeddings_sparse],
    'modelo9': [description_embeddings_sparse, reasonforchange_embeddings_sparse],
    'modelo10': [itndescimpact_embeddings_sparse, reasonforchange_embeddings_sparse],
    'modelo11': [description_embeddings_sparse],
    'modelo12': [itndescimpact_embeddings_sparse],
    'modelo13': [reasonforchange_embeddings_sparse],
    'modelo14': [encoded_categorical_sparse]
}

# Variable objetivo
y = data[target_col]
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Diccionario para almacenar resultados
results = []

# 6. Train and Test
# Función para calcular accuracy proporcional
def proportional_accuracy(y_true, y_pred):
    classes, counts = np.unique(y_true, return_counts=True)
    accuracies = [accuracy_score(y_true[y_true == c], y_pred[y_true == c]) for c in classes]
    weights = counts / counts.sum()
    return np.average(accuracies, weights=weights)

# Función para calcular métricas por clase de forma robusta
def calculate_class_metrics(y_true, y_pred, class_names):
    recalls = np.zeros(len(class_names))
    f1_scores = np.zeros(len(class_names))

    # Calcular métricas solo para clases presentes
    present_classes = np.intersect1d(y_true, y_pred)
    for c in present_classes:
        recalls[c] = recall_score(y_true, y_pred, labels=[c], average=None)[0]
        f1_scores[c] = f1_score(y_true, y_pred, labels=[c], average=None)[0]

    return recalls, f1_scores


for model_name, features in tqdm(combinations.items(), desc="Evaluando modelos"):
    # Combinar características
    X = hstack(features) if len(features) > 1 else features[0]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    # Configurar modelo con barra de progreso
    n_estimators = 100  # Ajusta este valor según necesites
    clf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    verbose=2,  # 1: muestra progreso por árbol, 2: muestra tiempo por paso
    n_estimators=100
    )

    clf.fit(X_train, y_train)

    # Resto del código igual...
    y_pred = clf.predict(X_test)

    # Calcular métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    prop_accuracy = proportional_accuracy(y_test, y_pred)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Métricas por clase
    recalls, f1_scores = calculate_class_metrics(y_test, y_pred, label_encoder_y.classes_)

    # Construir diccionario de resultados
    result = {
        'modelo': model_name,
        'combinacion': ', '.join([
            'Categóricas' if f is encoded_categorical_sparse else
            'DESCRIPCION' if f is description_embeddings_sparse else
            'ITNDESCIMPACT' for f in features]),
        'accuracy': accuracy,
        'accuracy_proporcional': prop_accuracy,
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
    }

    # Añadir métricas por clase
    for i, class_name in enumerate(label_encoder_y.classes_):
        result[f'recall_{class_name}'] = recalls[i]
        result[f'f1_{class_name}'] = f1_scores[i]

    # Guardar modelo
    model_path = f'models/{model_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    results.append(result)

# 7. Save Metrics
# Reordenar columnas (ahora sabemos que existen)
cols_order = ['modelo', 'combinacion', 'accuracy', 'accuracy_proporcional',
       'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1',
       'recall_High', 'f1_High', 'recall_Low', 'f1_Low', 'recall_Medium',
       'f1_Medium', 'recall_Null', 'f1_Null', 'recall_Other', 'f1_Other']

# results to df
results_df = pd.DataFrame(results)

results_df = results_df[cols_order]

# Save results
results_df.to_csv('/fhome/amir/TFG/code/data_quality/results/results_comb2.csv', index=False)

print("Results saved to results_comb2.csv")