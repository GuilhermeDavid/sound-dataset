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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import tensorflow_hub as hub
import librosa
from tqdm import tqdm
import random
import kagglehub

# ---------------------------------------------------------------
# 1. DOWNLOAD DO DATASET ---------------------------------------------------------------
path = kagglehub.dataset_download("soumendraprasad/musical-instruments-sound-dataset")
print("Path to dataset files:", path)

# ---------------------------------------------------------------
# 2. CONFIGURAÇÕES INICIAIS
# ---------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 5  # Aumentado para 5 segundos
BATCH_SIZE = 16
NUM_CLASSES = 4
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------
# 3. PATHS - baseado no path dinâmico do kagglehub
# ---------------------------------------------------------------
train_csv = os.path.join(path, "Metadata_Train.csv")
test_csv = os.path.join(path, "Metadata_Test.csv")
train_audio_path = os.path.join(path, "Train_submission", "Train_submission")
test_audio_path = os.path.join(path, "Test_submission", "Test_submission")

# ---------------------------------------------------------------
# 4. FUNÇÃO PARA CARREGAR ÁUDIO
# ---------------------------------------------------------------
def load_audio(file_path, duration=DURATION):
    """Carrega áudio mono, sample rate fixo e pad se necessário"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=duration)
    if len(y) < SAMPLE_RATE * duration:
        y = np.pad(y, (0, SAMPLE_RATE*duration - len(y)), mode="constant")
    return y

# ---------------------------------------------------------------
# 5. AUGMENTATION SIMPLES
# ---------------------------------------------------------------
def augment_audio(y):
    """Aplica augmentações simples no áudio"""
    # 1) Adicionar ruído branco
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise

    # 2) Time shifting (deslocamento no tempo)
    shift = int(np.random.uniform(-0.1, 0.1) * SAMPLE_RATE)  # shift até 0.1s
    y_shift = np.roll(y, shift)
    if shift > 0:
        y_shift[:shift] = 0
    else:
        y_shift[shift:] = 0

    # 3) Pitch shifting (mudança de tom)
    y_pitch = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=np.random.uniform(-2, 2))

    # Randommente escolhe uma das três augmentações
    return random.choice([y_noise, y_shift, y_pitch])

# ---------------------------------------------------------------
# 6. PREPARAR DATASET COM AUGMENTATION E BALANCEAMENTO
# ---------------------------------------------------------------
def prepare_dataset(csv_file, audio_folder, augment=True, balance=True):
    df = pd.read_csv(csv_file)
    
    # Mapeamento de classes para índices
    instrument_map = {}
    for idx, class_name in enumerate(sorted(df['Class'].unique())):
        instrument_map[class_name] = idx

    print("Mapeamento de classes:", instrument_map)
    
    # Carregar arquivos por classe
    data = {cls: [] for cls in instrument_map.keys()}
    missing_files = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processando áudios"):
        file_name = row['FileName']
        class_name = row['Class']
        file_path = os.path.join(audio_folder, file_name)
        if not os.path.exists(file_path):
            missing_files += 1
            print(f"Aviso: arquivo não encontrado {file_path}")
            continue
        audio = load_audio(file_path)
        data[class_name].append(audio)
    
    print(f"Arquivos faltantes: {missing_files}")

    # Balancear e aplicar augmentation
    if balance:
        max_count = max(len(audios) for audios in data.values())
    else:
        max_count = None  # Sem balanceamento

    X, y = [], []
    for class_name, audios in data.items():
        count = len(audios)
        audios_copy = audios.copy()
        # Se balancear, repetir e augmentar para atingir max_count
        if balance:
            while len(audios_copy) < max_count:
                audio_to_augment = random.choice(audios)
                augmented_audio = augment_audio(audio_to_augment) if augment else audio_to_augment
                audios_copy.append(augmented_audio)
        # Agora adiciona os áudios (originais + augmentados)
        for audio in audios_copy:
            X.append(audio)
            y.append(instrument_map[class_name])

    return np.array(X), np.array(y), instrument_map

# ---------------------------------------------------------------
# 7. CARREGAR DATASETS
# ---------------------------------------------------------------
print("Preparando dataset de treino com augmentation e balanceamento...")
X_train, y_train, instrument_map = prepare_dataset(train_csv, train_audio_path, augment=True, balance=True)

print("Preparando dataset de teste (sem augmentation)...")
X_test, y_test, _ = prepare_dataset(test_csv, test_audio_path, augment=False, balance=False)

# Divisão treino/validação
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
)

# ---------------------------------------------------------------
# 8. TRANSFER LEARNING COM YAMNet
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
# 9. NORMALIZAÇÃO DAS EMBEDDINGS
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_emb = scaler.fit_transform(X_train_emb)
X_val_emb = scaler.transform(X_val_emb)
X_test_emb = scaler.transform(X_test_emb)

# ---------------------------------------------------------------
# 10. CRIAÇÃO DO MODELO
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
# 11. TREINAMENTO
# ---------------------------------------------------------------
EPOCHS = 50
history = model.fit(
    X_train_emb, y_train,
    validation_data=(X_val_emb, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ---------------------------------------------------------------
# 12. GRÁFICOS DE LOSS E ACCURACY
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
# 13. MATRIZ DE CONFUSÃO
# ---------------------------------------------------------------
y_pred = np.argmax(model.predict(X_test_emb), axis=1)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                              display_labels=list(instrument_map.keys()))
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão - Classificação de Instrumentos (com melhorias)")
plt.show()
