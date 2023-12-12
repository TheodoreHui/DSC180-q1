import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

class Word2VecLabelGenerator:
    def __init__(self):
        self.lem = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.seed_vecs = None

    def lemmatize_sentence(self, sentence):
        """
        Lemmatizes the given sentence, removing English stopwords.

        Parameters:
        - sentence (str): Input sentence to lemmatize.

        Returns:
        - list of str: List of lemmatized tokens.
        """
        tokens = word_tokenize(sentence)
        lemmatized_tokens = [self.lem.lemmatize(token) for token in tokens if token not in self.stop_words]
        return lemmatized_tokens

    def load_pickle(self, path):
        """
        Loads a DataFrame from a pickle file and lemmatizes the 'sentence' column.

        Parameters:
        - path (str): Path to the pickle file.

        Returns:
        - pd.DataFrame: DataFrame with lemmatized 'sentence' column.
        """
        with open(path, 'rb') as file:
            df = pickle.load(file)
            df['sentence'] = df['sentence'].apply(self.lemmatize_sentence)
            return df

    def load_json(self, path):
        """
        Loads a DataFrame from a JSON file and lemmatizes the 'sentence' column.

        Parameters:
        - path (str): Path to the JSON file.

        Returns:
        - pd.DataFrame: DataFrame with lemmatized 'sentence' column.
        """
        with open(path, 'r') as file:
            return pd.read_json(file.read()).map(self.lem.lemmatize)

    def train_model(self, data):
        """
        Trains the Word2Vec model using the provided data and saves it to a binary file.

        Parameters:
        - data (pd.DataFrame): DataFrame containing 'sentence' column for training.

        Returns:
        - None
        """
        self.model = Word2Vec(sentences=data['sentence'], sg = 1, vector_size=256, window=8, min_count=1, workers=16, ns_exponent=1.1, sample=1e-3)
        self.model.save("word2vec_model.bin")


    def load_model(self, model_path):
        """
        Loads a pre-trained Word2Vec model from a specified path.

        Parameters:
        - model_path (str): Path to the pre-trained Word2Vec model.

        Returns:
        - None
        """
        self.model = Word2Vec.load(model_path)

    def seed_vectors(self, seeds):
        """
        Generates seed vectors based on the trained Word2Vec model.

        Parameters:
        - seeds (dict): Dictionary containing seed words for each category.

        Returns:
        - None
        """
        self.seed_vecs = {col: sum([self.model.wv[seed] for seed in seeds[col]]) / len(seeds[col]) for col in seeds}

    def doc_vectors(self, tokens):
        """
        Generates document vectors based on the trained Word2Vec model, handling missing tokens.

        Parameters:
        - tokens (list of list of str): List of tokenized sentences.

        Returns:
        - list of np.ndarray: List of document vectors.
        """
        vectorized_docs = []

        for doc in tokens:
            sum = 0
            for token in doc:
                try: 
                    sum += self.model.wv[token]
                except KeyError:
                    pass
            vectorized_docs.append(sum/len(doc))
        return vectorized_docs

    def cos_sim(self, v1, v2):
        """
        Computes cosine similarity between two vectors.

        Parameters:
        - v1 (np.ndarray): First vector.
        - v2 (np.ndarray): Second vector.

        Returns:
        - float: Cosine similarity between v1 and v2.
        """
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    def compute_relevance(self, doc_vec):
        """
        Computes relevance of document vectors to seed vectors.

        Parameters:
        - doc_vec (list of np.ndarray): List of document vectors.

        Returns:
        - list: List of predicted labels based on relevance.
        """
        out = []
        for doc in doc_vec:
            relevance = {seed_vec[0]: self.cos_sim(seed_vec[1], doc) for seed_vec in self.seed_vecs.items()}
            out.append(max(zip(relevance.values(), relevance.keys()))[1])
        return out

    def run_experiment(self, data):
        """
        Runs and scores model on new data

        Parameters:
        - data (pd.DataFrame): DataFrame containing 'sentence' and 'label' columns.

        Returns:
        - tuple: F1 macro and micro scores for the experiment.
        """
        data = data.reset_index(drop=True)
        doc_vec = self.doc_vectors(data['sentence'])
        data['pred_label'] = self.compute_relevance(doc_vec)
        return f1_score(data['label'], data['pred_label'], average='macro'), f1_score(data['label'], data['pred_label'], average='micro')