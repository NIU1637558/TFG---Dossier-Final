
from deep_translator import GoogleTranslator
from langdetect import detect
from pd import pandas

print('Reading data...')
data3 = pd.read_csv("/fhome/amir/TFG/data/CH_Total_label2.csv")

data3['ITNDESCIMPACT_trad']=False
data3['DESCRIPTION_trad']=False
data3['REASONFORCHANGE_trad']=False

# Función para detectar el idioma y traducir si es portugués
def translate_to_spanish(text):
    if isinstance(text, str):  # Asegurarse de que el texto sea una cadena
        try:
            # Detectar el idioma del texto
            detected_lang = detect(text)
            # Si el texto está en portugués, traducirlo al español
            if detected_lang == 'pt':
                translated = GoogleTranslator(source='pt', target='es').translate(text)
                return translated
        except Exception as e:
            return text
    return text  # Devolver el texto original si no es portugués o hay un error

# Aplicar la función de traducción a la columna ITNDESCIMPACT
print("Translating ITNDESCIMPACT")
data3['ITNDESCIMPACT_trad'] = data3['ITNDESCIMPACT'].apply(translate_to_spanish)

print("Translating DESCRIPTION")
data3['DESCRIPTION_trad'] = data3['DESCRIPTION'].apply(translate_to_spanish)

print("Translating REASONFORCHANGE")
data3['REASONFORCHANGE_trad'] = 
data3['REASONFORCHANGE'].apply(translate_to_spanish)

# Verificar algunos registros traducidos
print("Ejemplo de registros traducidos:")
print(data3[['ITNDESCIMPACT_trad']].head())