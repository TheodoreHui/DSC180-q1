import pickle
import json
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('wordnet')
nltk.download('punkt')

class TFIDFLabelGenerator:
    def __init__(self):
        self.lem = WordNetLemmatizer()
        self.seed = None
        self.data = None
        self.tfidf = TfidfVectorizer()
        self.matrix = None
        self.trim = 0
    
    def train(self, data, seed):
        """
        Train the TFIDFLabelGenerator with the provided data and seed words.

        Parameters:
        - data: DataFrame containing 'sentence' and 'label' columns
        - seed: Dictionary of seed words for different labels
        """
        self.seed = seed
        self.data = data.reset_index(drop=True)
        self.matrix = self.tfidf.fit(self.data['sentence'])
        self.trim = self.tune_imputation(self.generate_labels(self.data), self.data)[0]

    def lemmatize_sentence(self, sentence):
        """
        Lemmatize a sentence to simplify the corpus.

        Parameters:
        - sentence: Input sentence

        Returns:
        - Lemmatized sentence
        """
        tokens = word_tokenize(sentence)
        lemmatized_tokens = [self.lem.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def load_pickle(self, path):
        """
        Load a DataFrame from a pickle file and lemmatize the 'sentence' column.

        Parameters:
        - path: Path to the pickle file

        Returns:
        - DataFrame with lemmatized 'sentence' column
        """
        with open(path, 'rb') as file:
            df = pickle.load(file)
            df['sentence'] = df['sentence'].apply(self.lemmatize_sentence)
            return df

    def load_json(self, path):
        """
        Load a DataFrame from a JSON file and lemmatize the 'sentence' column.

        Parameters:
        - path: Path to the JSON file

        Returns:
        - DataFrame with lemmatized 'sentence' column
        """
        with open(path, 'r') as file:
            return pd.read_json(file.read()).map(self.lem.lemmatize)

    def compute_relevance(self, document):
        """
        Compute the relevance of a document to different labels based on TF-IDF weights.

        Parameters:
        - document: Input document

        Returns:
        - Predicted label with the highest relevance
        """
        relevance = {label: 0 for label in self.seed.keys()}
        relevance['na'] = sys.float_info.min
        tokens = document.split()
        weights = self.tfidf.transform([document])
        for label, words in self.seed.items():
            for word in words:
                #search for all seed words in document
                if word in tokens:
                    index = self.tfidf.vocabulary_.get(word)
                    #add tfidf weight of word from document to relevence label
                    if index is not None:
                        relevance[label] += weights[0,index]

        return max(zip(relevance.values(), relevance.keys()))[1]

    def generate_labels(self, data):
        """
        Generate labels for a given dataset based on document relevance to seed words.

        Parameters:
        - data: DataFrame containing 'sentence' column

        Returns:
        - Predicted labels for the input data
        """
        out = []
        for index, row in data.iterrows():
            out.append(self.compute_relevance(row['sentence']))

        return pd.Series(out)

    def score(self, expt, pred):
        """
        Calculate F1 macro and micro scores.

        Parameters:
        - expt: True labels
        - pred: Predicted labels

        Returns:
        - Tuple containing F1 macro and micro scores
        """
        return f1_score(expt, pred, average='macro'), f1_score(expt, pred, average='micro')

    def impute(self, pred, trim=0):
        """
        Impute 'na' labels by randomly selecting from the distribution of non-'na' labels.

        Parameters:
        - pred: Predicted labels
        - trim: Number of categories to trim from the bottom of the distribution

        Returns:
        - Labels with imputed 'na' values
        """
        pred_label_dist = pred[pred != 'na'].value_counts()
        if trim:
            pred_label_dist = pred_label_dist[:-trim]
        impute_na = lambda x: np.random.choice(
            pred_label_dist.index, p=pred_label_dist.values / sum(pred_label_dist.values)) if x == 'na' else x
        return pred.apply(impute_na)

    def tune_imputation(self, pred, data):
        """
        Automatically find the best imputation trim level for the highest F1 score.

        Parameters:
        - pred: Predicted labels
        - data: DataFrame containing true labels

        Returns:
        - Tuple containing the best imputation trim level and corresponding F1 scores
        """
        pred_labels = len(pred[pred != 'na'].unique())
        scores = {}
        for i in range(pred_labels):
            scores[i] = self.score(self.impute(pred, i), data['label'])
        self.trim = max(scores.items(), key=lambda x: sum(x[1]))[0]
        return max(scores.items(), key=lambda x: sum(x[1]))

    def run_experiment(self, data):
        """
        Runs model on new data using training hyperparameters.

        Parameters:
        - data: DataFrame containing 'sentence' and 'label' columns

        Returns:
        - Tuple containing F1 macro and micro scores for the experiment
        """
        data = data.reset_index(drop = True)
        pred = self.generate_labels(data)
        return self.score(data['label'], self.impute(pred, self.trim))