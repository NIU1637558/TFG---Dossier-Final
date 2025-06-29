import numpy as np
import pandas as pd
import pickle  # Para guardar/cargar embeddings
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec  # Para el modelo Word2Vec
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder
)
from scipy.sparse import csr_matrix, hstack  # Para matrices dispersas
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split


##### AUTOENCODERS  #####
class TextAutoEncoder(nn.Module):
    def __init__(self, input_dim=3*384, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec
########################


##### w2v embeddings #####
def get_avg_embedding(text, model):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_W2V_embeddings(balanced_data, text_cols, og_data):
    # Cargar el modelo de Word2Vec
    nltk.download('punkt_tab')

    # Tokenize text
    tokenized_texts = []

    for col in text_cols:
        if col in og_data.columns:
            texts = og_data[col].fillna("").astype(str).tolist()
            tokenized_texts.extend([word_tokenize(text.lower()) for text in texts])

    # Embedding
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=32,
        window=5,
        min_count=2,
        workers=4
    )

    # get average embedding
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = np.array([get_avg_embedding(text, w2v_model) for text in texts])
            text_embeddings.append(embeddings)

    # Concatenar embeddings o crear array vacío si no hay columnas válidas
    all_text_embeddings = np.hstack(text_embeddings) if text_embeddings else np.zeros((len(balanced_data), 0))

    return all_text_embeddings

def get_BERT_embeddings(balanced_data, text_cols, device):
    # Preparar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # BERT models
    # 1- paraphrase-multilingual-MiniLM-L12-v2
    # 2- all-MiniLM-L6-v2
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

    # Generar embeddings por columna de texto
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                device=device,
                normalize_embeddings=True
            )
            text_embeddings.append(embeddings)

    # Concatenar embeddings o crear array vacío si no hay columnas válidas
    all_text_embeddings = np.hstack(text_embeddings) if text_embeddings else np.zeros((len(balanced_data), 0))

    return all_text_embeddings

def get_AE1w2v_embeddings(balanced_data, text_cols, device):
    # Ruta al modelo entrenado
    model_path = "/fhome/amir/TFG/code/embedder/results/models/AE1_W2V_model.pth"
    
    # Configuración del modelo (asegúrate de que las dimensiones coincidan con el entrenamiento)
    input_dim = 96  # Dimensión de salida de Word2Vec
    latent_dim = 10  # Dimensión latente del encoder
    
    # Cargar el modelo y los pesos del encoder
    model = TextAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Cargar el modelo de Word2Vec
    nltk.download('punkt')
    tokenized_texts = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            tokenized_texts.extend([word_tokenize(text.lower()) for text in texts])

    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=32,  # Dimensión del embedding de Word2Vec
        window=5,
        min_count=2,
        workers=4
    )

    # Generar embeddings por columna de texto
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = np.array([get_avg_embedding(text, w2v_model) for text in texts])
            
            # Pasar los embeddings por el encoder
            text_embeddings.append(embeddings)

    # Concatenar embeddings o devolver array vacío
    inputs = np.hstack(text_embeddings)

    # inputs to tensor
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        all_text_embeddings = model.encoder(inputs).cpu().numpy()

    return all_text_embeddings
###############

def get_AE1BERT_embeddings(balanced_data, text_cols, device):
    # Cargar el modelo SentenceTransformer (BERT multilingüe)
    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Ruta al encoder del AutoEncoder BERT
    model_path = "/fhome/amir/TFG/code/embedder/results/models/AE1_BERT60_model.pth"

    # Configuración del AutoEncoder (ajústalo si tu AE tiene otras dimensiones)
    input_dim = 384*3  # BERT output dimension
    latent_dim = 60  # Dimensión latente del encoder

    # Cargar el encoder del AutoEncoder
    model = TextAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Obtener embeddings BERT por columna de texto
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = bert_model.encode(
                texts,
                show_progress_bar=True,
                device=device,
                normalize_embeddings=True
            )

            text_embeddings.append(embeddings)

    # Concatenar embeddings o devolver array vacío
    inputs = np.hstack(text_embeddings)

    # inputs to tensor
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        all_text_embeddings = model.encoder(inputs).cpu().numpy()

    return all_text_embeddings

def get_AE1BERTtuned_embeddings(balanced_data, text_cols, device):
    # Cargar el modelo SentenceTransformer (BERT multilingüe)
    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Ruta al encoder del AutoEncoder BERT
    model_path = "/fhome/amir/TFG/code/embedder/results/models/AE1_tunedBERT60_model.pth"

    # Configuración del AutoEncoder (ajústalo si tu AE tiene otras dimensiones)
    input_dim = 384*3  # BERT output dimension
    latent_dim = 60  # Dimensión latente del encoder

    # Cargar el encoder del AutoEncoder
    model = TextAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Obtener embeddings BERT por columna de texto
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = bert_model.encode(
                texts,
                show_progress_bar=True,
                device=device,
                normalize_embeddings=True
            )

            text_embeddings.append(embeddings)

    # Concatenar embeddings o devolver array vacío
    inputs = np.hstack(text_embeddings)

    # inputs to tensor
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        all_text_embeddings = model.encoder(inputs).cpu().numpy()

    return all_text_embeddings


def get_D2V_embeddings(balanced_data, text_cols, og_data):
    # Cargar el modelo de Doc2Vec
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    # Tokenizar el texto
    tokenized_texts = []
    for col in text_cols:
        if col in og_data.columns:
            texts = og_data[col].fillna("").astype(str).tolist()
            tokenized_texts.extend([word_tokenize(text.lower()) for text in texts])

    # Crear documentos etiquetados para Doc2Vec
    tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokenized_texts)]

    # Entrenar el modelo Doc2Vec
    d2v_model = Doc2Vec(
        documents=tagged_data,
        vector_size=32,
        window=5,
        min_count=2,
        workers=4,
        epochs=100
    )

    # Obtener embeddings por columna de texto
    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = np.array([d2v_model.infer_vector(word_tokenize(text.lower())) for text in texts])
            text_embeddings.append(embeddings)

    # Concatenar embeddings o devolver array vacío
    all_text_embeddings = np.hstack(text_embeddings) if text_embeddings else np.zeros((len(balanced_data), 0))

    return all_text_embeddings

from sentence_transformers import SentenceTransformer

def get_DistilBERT_embeddings(balanced_data, text_cols, device):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2', device=device)
    model = model.to(device)

    text_embeddings = []
    for col in text_cols:
        if col in balanced_data.columns:
            texts = balanced_data[col].fillna("").astype(str).tolist()
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                device=device,
                normalize_embeddings=True
            )
            text_embeddings.append(embeddings)

    all_text_embeddings = np.hstack(text_embeddings) if text_embeddings else np.zeros((len(balanced_data), 0))

    return all_text_embeddings


from nltk.corpus import stopwords
import re 

def clean_stopwords(txt):
    stop_words = set(stopwords.words('spanish'))

    # Filter STOPWORDS
    tokens = word_tokenize(txt.lower())
    filtered_text = ' '.join([word for word in tokens if word not in stop_words])

    # Filter COMPONENTS
    text = str(filtered_text)
    text = re.sub(r'\[[A-Z0-9_\-\s]+\]', '', text)
    text = re.sub(r'\b(?:[A-Z0-9]+(?:[\s\-]+[A-Z0-9]+)+)\b', '', text)
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ0-9\s]', '', text)

    return text

def prepare_features(balanced_data,og_data, embedder, target_col = 'REJECTED'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #----- 1. prepare data -----------
    categorical_cols = [
        'STATUS', 'ITNCHGCOMCANAL', 'ITNESCENARIO',
        'ACTSTART', 'PMCHGTYPE', 'OWNERGROUP', 'PMCHGCONCERN',
        'ITDCLOSURECODE', 'PROBLEMCODE', 'ITDCHCREATEDBYGROUP',
        'ACTFINISH', 'FIRSTAPPRSTATUS', 'PMCHGAPPROVALSTATE',
        'ITNAMBITO'
    ]

    # # for DQ escenario
    # categorical_cols = [
    #     'STATUS', 'ITNCHGCOMCANAL',
    #     'ACTSTART', 'PMCHGTYPE', 'OWNERGROUP', 'PMCHGCONCERN',
    #     'ITDCLOSURECODE', 'PROBLEMCODE', 'ITDCHCREATEDBYGROUP',
    #     'ACTFINISH', 'FIRSTAPPRSTATUS', 'PMCHGAPPROVALSTATE',
    #     'ITNAMBITO'
    # ]

    # # for DQ impact
    # categorical_cols = [
    #     'STATUS', 'ITNCHGCOMCANAL',
    #     'ACTSTART', 'PMCHGTYPE', 'OWNERGROUP', 'PMCHGCONCERN',
    #     'ITDCLOSURECODE', 'PROBLEMCODE', 'ITDCHCREATEDBYGROUP',
    #     'ACTFINISH', 'FIRSTAPPRSTATUS', 'PMCHGAPPROVALSTATE',
    #     'ITNAMBITO'
    # ]

    text_cols = ['DESCRIPTION_trad', 'ITNDESCIMPACT_trad', 'REASONFORCHANGE_trad']
    numeric_cols = ['DESCRIPTION_len', 'ITNDESCIMPACT_len', 'REASONFORCHANGE_len']

    # # for dq impact eliminar ITNDESCIMPACT_len
    # text_cols = ['DESCRIPTION_trad','REASONFORCHANGE_trad']
    # numeric_cols = ['DESCRIPTION_len', 'REASONFORCHANGE_len']

    # Calcular las longitudes de las columnas de texto y guardarlas en numeric_cols
    for text_col, numeric_col in zip(text_cols, numeric_cols):
        if text_col in balanced_data.columns:
            balanced_data[numeric_col] = balanced_data[text_col].fillna("").astype(str).str.len()

    # Fill missing values
    for col in categorical_cols + text_cols:
        if col in balanced_data.columns:
            balanced_data[col] = balanced_data[col].fillna('')

    # ---- 2. Prepare embeddings W2V ----
    print("\nGenerating features...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if embedder == 'w2v':
        all_text_embeddings = get_W2V_embeddings(balanced_data, text_cols, og_data)

    elif embedder == 'd2v':
        all_text_embeddings = get_D2V_embeddings(balanced_data, text_cols, og_data)

    elif embedder == 'AE1w2v':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        all_text_embeddings = get_AE1w2v_embeddings(balanced_data, text_cols, device)

    elif embedder == 'BERT':
        all_text_embeddings = get_BERT_embeddings(balanced_data, text_cols, device)

    elif embedder == 'AE1BERT':
        all_text_embeddings = get_AE1BERT_embeddings(balanced_data, text_cols, device)

    elif embedder == 'distBERT':
        all_text_embeddings = get_DistilBERT_embeddings(balanced_data, text_cols, device)

    elif embedder == 'AE1BERTtuned':
        all_text_embeddings = get_AE1BERTtuned_embeddings(balanced_data, text_cols, device)
    
    elif embedder == None:
        all_text_embeddings = np.zeros((len(balanced_data), 0))
    
    else:
        raise ValueError(f"Embedder '{embedder}' is not recognized. Choose from: w2v, d2v, AE1w2v, BERT, AE1BERT, distBERT, AE1BERTtuned.")

    # Save embeddings
    with open('/fhome/amir/TFG/code/classificators/DL/embeddings/text_embeddings.pkl', 'wb') as f:
        pickle.dump(all_text_embeddings, f)

    # # Cargar embeddings
    # with open('/fhome/amir/TFG/code/classificators/DL/embeddings/text_embeddings.pkl', 'rb') as f:
    #     all_text_embeddings = pickle.load(f)
    # print("Embeddings loaded from text_embeddings.pkl")

    # ---- 3. Encode categorical features ----  
    label_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_categorical = label_encoder.fit_transform(balanced_data[categorical_cols])

    # ----- 4. Normalize all data ----
    num_scaler = StandardScaler()
    balanced_data[numeric_cols] = num_scaler.fit_transform(balanced_data[numeric_cols])
    cat_scaler = StandardScaler()
    encoded_categorical = cat_scaler.fit_transform(encoded_categorical)
    text_scaler = StandardScaler()

    # si tenim embedder, son data txt
    if embedder != None:
        all_text_embeddings = text_scaler.fit_transform(all_text_embeddings)
        txt_shape = all_text_embeddings.shape[1]

        # Combine features
        X = hstack([
            csr_matrix(all_text_embeddings.astype(np.float32)), 
            csr_matrix(encoded_categorical.astype(np.float32)),
            csr_matrix(balanced_data[numeric_cols].astype(np.float32).values)
        ])
        
    # si no tenim embedder, son data categorica
    else:
        txt_shape = 0
        X = hstack([
            csr_matrix(encoded_categorical.astype(np.float32)),
            csr_matrix(balanced_data[numeric_cols].astype(np.float32).values)
        ])

    y = balanced_data[target_col].values

    og_indices = balanced_data['original_index'].values
    
    return X, y, categorical_cols, text_cols, og_indices, txt_shape

def install_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')