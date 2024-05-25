from sentence_transformers import SentenceTransformer
import numpy as np
import random
from collections import defaultdict
import pandas as pd
class SimpleLSH:
    def __init__(self, input_dim, num_hash_tables=10, num_hash_bits=12):
        self.input_dim = input_dim
        self.num_hash_tables = num_hash_tables
        self.num_hash_bits = num_hash_bits
        self.hash_tables = [defaultdict(list) for _ in range(num_hash_tables)]
        self.random_vectors = [np.random.randn(self.num_hash_bits, input_dim) for _ in range(num_hash_tables)]

    def _hash(self, vector, random_vectors):
        projection = np.dot(random_vectors, vector)
        return tuple((projection > 0).astype(int))

    def add(self, vector, idx):
        for i in range(self.num_hash_tables):
            hash_value = self._hash(vector, self.random_vectors[i])
            self.hash_tables[i][hash_value].append(idx)

    def query(self, vector):
        candidates = set()
        for i in range(self.num_hash_tables):
            hash_value = self._hash(vector, self.random_vectors[i])
            candidates.update(self.hash_tables[i].get(hash_value, []))
        return candidates

class SemanticLabelingLSH:
    def __init__(self, model_name='Alibaba-NLP/gte-base-en-v1.5', num_hash_tables=10, num_hash_bits=12, threshold=0.75):
        self.model = SentenceTransformer(model_name,trust_remote_code=True)
        self.threshold = threshold
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.lsh = SimpleLSH(self.dimension, num_hash_tables, num_hash_bits)
        self.sentences = []
        self.labels = []
        self.next_label = 0

    def add_sentence(self, sentence):
        embedding = self.model.encode(sentence,normalize_embeddings=True)
        if len(self.sentences) == 0:
            self.sentences.append(embedding)
            self.lsh.add(embedding, 0)
            self.labels.append(self.next_label)
            self.next_label += 1
            return 0
        else:
            candidates = self.lsh.query(embedding)
            min_distance = float('inf')
            best_candidate = None
            for idx in candidates:
                distance = np.linalg.norm(embedding - self.sentences[idx])
                if distance < min_distance:
                    min_distance = distance
                    best_candidate = idx

            if best_candidate is not None and min_distance < self.threshold:
                label = self.labels[best_candidate]
            else:
                label = self.next_label
                self.next_label += 1
            
            self.sentences.append(embedding)
            self.lsh.add(embedding, len(self.sentences) - 1)
            self.labels.append(label)
            return label

tp1=SemanticLabelingLSH()
tp2=SemanticLabelingLSH()
df=pd.read_csv('point_made/point_made19.csv')
df["hasp"]=df["point_made"].apply(tp1.add_sentence)
df["hasq"]=df["ques_made"].apply(tp2.add_sentence)

# # Group by 'hasp' and 'hasq' and aggregate the text entries into lists
grp = df.groupby(['hasq','hasp'])[["Comment","point_made","ques_made"]].agg(list).reset_index()
# # Sort the grouped data by 'hasp' and 'hasq'

sorted_grouped_df = grp.sort_values(by=['hasq','hasp'])
for col in ["Comment","point_made","ques_made"]:
    sorted_grouped_df[col] = sorted_grouped_df[col].apply(lambda x: '\n'.join(x))
print(sorted_grouped_df)
sorted_grouped_df.to_csv('output/result.csv')
