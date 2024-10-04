# env ag
import pandas as pd
import warnings
from darts import TimeSeries
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from PIL import Image
import plotly.express as px
import os
import requests
import zipfile
import shutil

warnings.filterwarnings('ignore')


# Impostazioni Layout
st.set_page_config(layout="wide")
url_immagine = 'https://github.com/MarcelloGalimberti/Sentiment/blob/main/Ducati_red_logo.png?raw=true'

col_1, col_2 = st.columns([1, 3])
with col_1:
    st.image(url_immagine, width=150)
with col_2:
    st.title('Scheduling MSV4 MTO')

# Caricamento dati
uploaded_S = st.file_uploader("Carica S") # Dati storici mensili per marchio.xlsx
if not uploaded_S:
    st.stop()
df_dati=pd.read_excel(uploaded_S, parse_dates=True)

# Data re-shape per AutoGluon
df_long = df_dati.melt(id_vars=['Mese-anno'],  # Colonne da mantenere intatte
                       value_vars=['S1', 'S2', 'S3'],  # Colonne da impilare
                       var_name='item_id',  # Nome della nuova colonna che conterrà i nomi delle colonne originali (S1, S2, S3)
                       value_name='target')  # N

# Preparazione training e test
ultimo_mese = df_long['Mese-anno'].max()
data_split = ultimo_mese - pd.DateOffset(months=6) # qui mettere input da Streamlit
df_dati_train = df_dati.loc[df_dati['Mese-anno'] < data_split]
df_dati_test = df_dati.loc[df_dati['Mese-anno'] >= data_split]

train = df_dati_train.melt(id_vars=['Mese-anno'],   
                     value_vars=['S1', 'S2', 'S3'],   
                     var_name='item_id',   
                     value_name='target')   

test = df_dati_test.melt(id_vars=['Mese-anno'],   
                    value_vars=['S1', 'S2', 'S3'],   
                    var_name='item_id',   
                    value_name='target')

train_data = TimeSeriesDataFrame.from_data_frame(
    train,
    id_column="item_id",
    timestamp_column="Mese-anno"
)

test_data = TimeSeriesDataFrame.from_data_frame(
    test,
    id_column="item_id",
    timestamp_column="Mese-anno"
)

all_data = TimeSeriesDataFrame.from_data_frame(
    df_long,
    id_column="item_id",
    timestamp_column="Mese-anno"
)

# Grafico
fig = px.line(df_long, x='Mese-anno', y='target', color='item_id')
st.plotly_chart(fig, use_container_width=True)

####################################################################



# Carico modello da locale
#percorso_modello = '/Users/marcello_galimberti/Documents/Python Scripts/dart/modelli_ag'
#predictor = TimeSeriesPredictor.load(percorso_modello)
#df_leaderboard = predictor.leaderboard(all_data)
#st.dataframe(df_leaderboard)
#st.stop()

########
# Carico zip da GitHub
# ora scarica i file in model/modelli_ag e lista directory e file (da eliminare in seguito)


# URL del repository GitHub in cui hai salvato il modello
MODEL_REPO_URL = 'https://raw.githubusercontent.com/MarcelloGalimberti/S_Forecasting/main/modelli_ag.zip'
MODEL_DIR = './model'  # Directory dove scaricare il modello

# Funzione per scaricare e decomprimere il modello
def download_model(repo_url, model_dir):
    zip_path = os.path.join(model_dir, 'model.zip')
    
    try:
        # Scarica il file zip del repository
        st.write("Scaricando il modello da GitHub...")
        response = requests.get(repo_url)

        # Verifica che il file sia stato scaricato correttamente
        if response.status_code != 200:
            st.error(f"Errore nel download del file: {response.status_code}")
            return False

        # Scrivi il contenuto del file scaricato
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        # Verifica che sia un file zip valido
        if not zipfile.is_zipfile(zip_path):
            st.error("Il file scaricato non è un file ZIP valido.")
            os.remove(zip_path)
            return False

        st.write(f"File ZIP scaricato correttamente: {zip_path}")

        # Estrai il file zip nella directory del modello
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            st.write("Inizio dell'estrazione...")
            zip_ref.extractall(model_dir)
            st.write("Estrazione completata!")     

        # Mostra il contenuto della directory estratta
        #st.write("Contenuto della directory estratta:")
        #for root, dirs, files in os.walk(model_dir):
        #    st.write(f"Root: {root}")
        #    st.write(f"Directories: {dirs}")
        #    st.write(f"Files: {files}")

        return True

    except zipfile.BadZipFile as e:
        st.error(f"Errore durante l'estrazione del file ZIP: {e}")
        return False

    except Exception as e:
        st.error(f"Errore durante il download o l'estrazione del modello: {e}")
        return False


# Scarica il modello solo se non esiste già nella directory
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)
    success = download_model(MODEL_REPO_URL, MODEL_DIR)
    if not success:
        st.write('Modello non caricato')
        st.stop()  # Interrompi l'esecuzione in caso di errore

# Interfaccia Streamlit
st.title("Previsione vendite con AutoGluon")

# Carica il modello da AutoGluon
def load_autogluon_model(model_dir):
    st.write("Caricando il modello AutoGluon...")
    #predictor = TabularPredictor.load(model_dir)
    predictor = TimeSeriesPredictor.load(model_dir)
    return predictor

dir_locale = './model/modelli_ag'

# Carica il modello
predictor_caricato = load_autogluon_model(dir_locale)

df_leaderboard = predictor_caricato.leaderboard(all_data)
st.dataframe(df_leaderboard)

st.stop()


# Carica il modello (se l'estrazione è riuscita)
predictor = load_autogluon_model(MODEL_DIR)

# Esempio di input per la previsione
st.write("Inserisci i dati per la previsione:")
# Supponiamo che i dati di input siano già formattati correttamente per il modello
# Aggiungi il tuo form per l'inserimento dei dati qui

# Esegui la previsione
if st.button("Prevedi"):
    # Esempio di input (deve essere personalizzato per il tuo modello e i tuoi dati)
    input_data = {'feature1': 10, 'feature2': 20}
    prediction = predictor.predict(input_data)
    
    st.write("Previsione:", prediction)






# alternativa: leggere file da locale