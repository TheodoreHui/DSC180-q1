import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.downloader as api

class Word2VecLabelGenerator:
    def __init__(self):
        self.lem = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.seed_vecs = None

    def lemmatize_sentence(self, sentence):
        tokens = word_tokenize(sentence)
        lemmatized_tokens = [self.lem.lemmatize(token) for token in tokens if token not in self.stop_words]
        return lemmatized_tokens

    def load_pickle(self, path):
        with open(path, 'rb') as file:
            df = pickle.load(file)
            df['sentence'] = df['sentence'].apply(self.lemmatize_sentence)
            return df

    def load_json(self, path):
        with open(path, 'r') as file:
            return pd.read_json(file.read()).map(self.lem.lemmatize)

    def train_model(self, data):
        self.model = Word2Vec(sentences=data['sentence'], sg = 1, vector_size=256, window=8, min_count=1, workers=16, ns_exponent=1.1, sample=1e-3)
        self.model.save("word2vec_model.bin")


    def load_model(self, model_path):
        self.model = Word2Vec.load(model_path)

    def seed_vectors(self, seeds):
        self.seed_vecs = {col: sum([self.model.wv[seed] for seed in seeds[col]]) / len(seeds[col]) for col in seeds}

    def doc_vectors(self, tokens):
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
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    def compute_relevance(self, doc_vec):
        out = []
        for doc in doc_vec:
            relevance = {seed_vec[0]: self.cos_sim(seed_vec[1], doc) for seed_vec in self.seed_vecs.items()}
            out.append(max(zip(relevance.values(), relevance.keys()))[1])
        return out

    def run_experiment(self, data):
        data = data.reset_index(drop=True)
        doc_vec = self.doc_vectors(data['sentence'])
        data['pred_label'] = self.compute_relevance(doc_vec)
        return f1_score(data['label'], data['pred_label'], average='macro'), f1_score(data['label'], data['pred_label'], average='micro')