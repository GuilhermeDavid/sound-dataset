# ---------------------------------------------------------------
# CLASSIFICAÇÃO DE ÁUDIOS - MUSICAL INSTRUMENT'S SOUND DATASET
# ---------------------------------------------------------------
# Objetivo: Classificar os sons de instrumentos (Guitarra, Bateria, Violino, Piano)
# Usando Transfer Learning com YAMNet e TensorFlow
# ---------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow_hub as hub
import librosa
from tqdm import tqdm
import kagglehub

# ---------------------------------------------------------------
# 1. DOWNLOAD DO DATASET
# ---------------------------------------------------------------
path = kagglehub.dataset_download("soumendraprasad/musical-instruments-sound-dataset")
print("Path to dataset files:", path)

# ---------------------------------------------------------------
# 2. CONFIGURAÇÕES INICIAIS
# ---------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 3  # segundos por áudio
BATCH_SIZE = 16
NUM_CLASSES = 4

# Caminhos para os CSVs e pastas de áudio
train_csv = os.path.join(path, "Metadata_Train.csv")
test_csv = os.path.join(path, "Metadata_Test.csv")
train_audio_path = os.path.join(path, "Train_submission", "Train_submission")
test_audio_path = os.path.join(path, "Test_submission", "Test_submission")

# ---------------------------------------------------------------
# 3. FUNÇÃO PARA CARREGAR ÁUDIO
# ---------------------------------------------------------------
def load_audio(file_path, duration=DURATION):
    """Carrega áudio mono, sample rate fixo e pad se necessário"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=duration)
    if len(y) < SAMPLE_RATE * duration:
        y = np.pad(y, (0, SAMPLE_RATE*duration - len(y)), mode="constant")
    return y

# ---------------------------------------------------------------
# 4. PREPARAR DATASET A PARTIR DO CSV
# ---------------------------------------------------------------
def prepare_dataset(csv_file, audio_folder):
    df = pd.read_csv(csv_file)
    X, y = [], []
    
    # Mapeamento de classes para índices
    instrument_map = {}
    for idx, class_name in enumerate(sorted(df['Class'].unique())):
        instrument_map[class_name] = idx
    
    print("Mapeamento de classes:", instrument_map)
    
    missing_files = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processando áudios"):
        file_name = row['FileName']
        class_name = row['Class']
        file_path = os.path.join(audio_folder, file_name)
        if not os.path.exists(file_path):
            missing_files += 1
            print(f"Aviso: arquivo não encontrado {file_path}")
            continue
        X.append(load_audio(file_path))
        y.append(instrument_map[class_name])
    
    print(f"Arquivos faltantes: {missing_files}")
    return np.array(X), np.array(y), instrument_map

# ---------------------------------------------------------------
# 5. CARREGAR DATASETS
# ---------------------------------------------------------------
print("Preparando dataset de treino...")
X_train, y_train, instrument_map = prepare_dataset(train_csv, train_audio_path)
print("Preparando dataset de teste...")
X_test, y_test, _ = prepare_dataset(test_csv, test_audio_path)

# Divisão treino/validação
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# ---------------------------------------------------------------
# 6. TRANSFER LEARNING COM YAMNet
# ---------------------------------------------------------------
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

def extract_embedding(audio):
    tensor_audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet_model(tensor_audio)
    return tf.reduce_mean(embeddings, axis=0)

def preprocess_dataset(audio_list):
    embeddings = []
    for audio in tqdm(audio_list, desc="Extraindo embeddings"):
        emb = extract_embedding(audio)
        embeddings.append(emb.numpy())
    return np.array(embeddings)

print("Extraindo embeddings do treino...")
X_train_emb = preprocess_dataset(X_train)
print("Extraindo embeddings da validação...")
X_val_emb = preprocess_dataset(X_val)
print("Extraindo embeddings do teste...")
X_test_emb = preprocess_dataset(X_test)

# ---------------------------------------------------------------
# 7. CRIAÇÃO DO MODELO
# ---------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(1024,)),  # YAMNet embedding size
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------------------------------------------------------------
# 8. TREINAMENTO
# ---------------------------------------------------------------
EPOCHS = 50
history = model.fit(
    X_train_emb, y_train,
    validation_data=(X_val_emb, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ---------------------------------------------------------------
# 9. GRÁFICOS DE LOSS E ACCURACY
# ---------------------------------------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Evolução da Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ---------------------------------------------------------------
# 10. MATRIZ DE CONFUSÃO
# ---------------------------------------------------------------
y_pred = np.argmax(model.predict(X_test_emb), axis=1)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                              display_labels=list(instrument_map.keys()))
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão - Classificação de Instrumentos")
plt.show()
