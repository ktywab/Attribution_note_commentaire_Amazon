
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
import zipfile
import joblib

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

#Importation du modèle BERTDISTIL (variante très léger de BERT) sur Hugging Face
from transformers import (
    DistilBertTokenizer,#Tokenizer pour DistilBert
    TFDistilBertModel,#Model DistilBert
    DataCollatorWithPadding,#Gestion du padding pour avoir les sequence de même taille
    BertTokenizer,#Model Bert
    TFBertModel#Model Bert
)

#Outils pour le suivi et la visualisation :
from tqdm.notebook import tqdm #Pour afficher barre de progression dans jupyter notebook
import itertools #Pour gérer les itérations
import multiprocessing #Pour faire de la parallélisation
import os

np.random.seed(987654321)
tf.random.set_seed(987654321)



vocab_size = 30000 #Taille max du dictionnaire pour la tokenisation ici limité aux 30 000 mots les plus fréquents dans les données d'entraînement
hide_most_frequently = 0 #ignoré les mots les plus fréquent comme "le, la , ..." qui sont peu informatif dans certaines contexte
#ici 0 nous permet de rien ignorer
review_len = 512 #fixe la longueur maximale des séquences de texte traitées ici 512 tokens, la limite standard pour BERT.

epochs = 20
batch_size = 50

fit_verbosity = 1
scale = 1



## Création de classe héritière de modèle Keras
class ClassificationModel(keras.Model):

    def __init__(self, bert_model):
        super(ClassificationModel, self).__init__()
        self.bert_model = bert_model
        """
        - pre_classifier : Une couche dense avec 768 unités et une activation relu, qui sert à transformer la sortie de BERT en une représentation adaptée pour la classification.
        - Une couche Dropout avec un taux de 0.1, utilisée pour éviter le surapprentissage en désactivant aléatoirement 10% des neurones lors de l'entraînement.
        - classifier : Une couche dense finale avec 2 unités (pour une classification binaire) et sans activation. La sortie brute de cette couche (logits) est interprétée comme les scores pour chaque classe.
        """
        self.pre_classifier = Dense(1000, activation='relu')
        self.dropout = Dropout(0.1)
        self.classifier = Dense(5)#5 pour le nombre de classe à prédire

    # Methode
    def call(self, x):
      """
      - Passage par BERT produisant une sortie last_hidden_state.
      - Extraction du token [CLS] : x[:, 0]
      - pre_classifier : La couche dense avec relu affine la représentation [CLS].
      - dropout : La couche Dropout désactive une fraction des neurones pour renforcer la robustesse du modèle.
      - classifier : La dernière couche dense produit deux logits (non normalisés), un pour chaque classe
      """
      x = self.bert_model(x)
      x = x.last_hidden_state#Pour recupérrer toute la liste d'embedding aocrrespondantt au token de chaque phrase dans l'espace latent de compréhension du modèle
      x = x[:, 0] # récupère la représentation du premier token [CLS], qui résume le sens général de la séquence (car BERT bidirectionnelle).
      x = self.pre_classifier(x)
      x = self.dropout(x)
      x = self.classifier(x)
      return x

       # Méthode de prédiction avec probabilités pour 5 classes
    def predict(self, texts):
      """
      Prédit la classe d'une liste de phrases avec probabilités.

      Args:
      texts (list): Liste de phrases à analyser.

      Returns:
      list: Liste de tuples contenant la classe prédite et la probabilité associée.
      """
      # Tokenisation des textes
      tokenized_texts = [tokenize_sample(text) for text in texts]
      collated = data_collator(tokenized_texts)

      # Convertir les données en tenseurs TensorFlow
      inputs = {
          'input_ids': tf.convert_to_tensor(collated['input_ids']),
          'attention_mask': tf.convert_to_tensor(collated['attention_mask'])
      }

      # Effectuer les prédictions
      logits = self(inputs)
      probs = tf.nn.softmax(logits, axis=-1)  # Calcule les probabilités pour chaque classe
      predictions = tf.argmax(probs, axis=-1)

      # Conversion des prédictions en étiquettes lisibles (ajustez les étiquettes à vos classes)
      class_labels = ["1", "2", "3", "4", "5"]

      # Formatage des résultats avec étiquette et probabilité
      results = [
          (class_labels[pred], float(probs[i][pred]))
          for i, pred in enumerate(predictions)
      ]

      return results
