# Classificação de Áudios - Musical Instrument's Sound Dataset 🎵

Este projeto realiza a **classificação de sons de instrumentos musicais** (Guitarra, Bateria, Piano e Violino) usando **Transfer Learning** com o modelo **YAMNet** do TensorFlow.  
O objetivo é extrair embeddings dos áudios e treinar uma rede neural para identificar o instrumento presente no som.

---

## Estrutura do Projeto

```
├── Atividade.py              # Código principal do projeto
├── requirements.txt          # Dependências do projeto
└── README.md                 # Este arquivo
```

---

## Ferramentas e Tecnologias Utilizadas

- **Python 3.10**
- **TensorFlow 2.x**
- **TensorFlow Hub** (YAMNet)
- **Librosa** (Processamento de áudio)
- **NumPy / Pandas** (Manipulação de dados)
- **Matplotlib / Seaborn** (Visualização de resultados)
- **scikit-learn** (Divisão de dataset e métricas)
- **tqdm** (Barra de progresso)
- **KaggleHub** (Download de datasets diretamente do Kaggle)

---

## Funcionalidades do Código

1. **Download do dataset** do Kaggle usando `kagglehub`.
2. **Pré-processamento dos áudios**: conversão para mono, sample rate fixo, e padding para duração uniforme.
3. **Leitura dos CSVs de metadados** para mapear arquivos aos instrumentos corretos.
4. **Extração de embeddings de áudio** com o modelo pré-treinado YAMNet.
5. **Treinamento de rede neural** sobre embeddings para classificar instrumentos.
6. **Visualização dos resultados**: gráficos de loss, accuracy e matriz de confusão.

---

## Como Rodar o Projeto

1. **Clonar o repositório**

2. **Instalar dependências**

```bash
pip install -r requirements.txt
```

3. **Executar o código principal**

```bash
python Atividade.py
```

> ⚠️ **Observação:** Alguns arquivos de áudio podem não estar presentes no dataset. O código irá pular esses arquivos e exibir avisos, mas continuará a execução normalmente.

---

## Resultados Esperados

- Treinamento da rede neural com validação cruzada.
- Gráficos de evolução de **Loss** e **Accuracy**.
- Matriz de confusão mostrando a performance do modelo na classificação dos instrumentos.


