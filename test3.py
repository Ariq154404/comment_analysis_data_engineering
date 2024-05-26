from sentence_transformers import SentenceTransformer
import numpy as np
import random
import pandas as pd 
np.random.seed(42)
class SimpleLSH:
    def __init__(self, input_dim, num_hash_tables=2, num_hash_bits=3):
        self.input_dim = input_dim
        self.num_hash_tables = num_hash_tables
        self.num_hash_bits = num_hash_bits
        self.random_vectors = [np.random.randn(self.num_hash_bits, input_dim) for _ in range(num_hash_tables)]

    def _hash(self, vector):
        projectioncs = []
        for i in range(self.num_hash_tables):
            projection = np.dot(self.random_vectors[i], vector)
            binary_array = (projection > 0).astype(int)
            binary_string = ''.join(binary_array.astype(str))
            projections.append(int(binary_string, 2))
        return tuple(projections)

   

class SemanticLabelingLSH:
    def __init__(self, model_name='paraphrase-distilroberta-base-v1', num_hash_tables=2, num_hash_bits=3):
        self.model = SentenceTransformer(model_name,trust_remote_code=True)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.lsh = SimpleLSH(self.dimension, num_hash_tables, num_hash_bits)
      

    def gen_hash(self, sentence):
        embedding = self.model.encode(sentence)
        return self.lsh._hash(embedding)

tp1=SemanticLabelingLSH()
tp2=SemanticLabelingLSH()
df=pd.read_csv('point_made/point_made19.csv')
df["hasp"]=df["point_made"].apply(tp1.gen_hash)
df["hasq"]=df["ques_made"].apply(tp2.gen_hash)

# # Group by 'hasp' and 'hasq' and aggregate the text entries into lists
grp = df.groupby(['hasq','hasp'])[["Comment","point_made","ques_made"]].agg(list).reset_index()
# # Sort the grouped data by 'hasp' and 'hasq'

sorted_grouped_df = grp.sort_values(by=['hasq','hasp'])
for col in ["Comment","point_made","ques_made"]:
    sorted_grouped_df[col] = sorted_grouped_df[col].apply(lambda x: '\n'.join(x))
print(sorted_grouped_df)
sorted_grouped_df.to_csv('output/result.csv')
