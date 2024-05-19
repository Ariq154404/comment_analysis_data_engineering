from langchain_community.llms import Ollama
from langgraph.graph import END, Graph
import pandas as pd
import re
#from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow_hub as hub
from sklearn.preprocessing import normalize

llm = Ollama(model="phi3")
#model = SentenceTransformer('all-MiniLM-L6-v2')
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)
np.random.seed(79)

# kmeans = MiniBatchKMeans(n_clusters=11, random_state=0)
class LSH:
    def __init__(self, num_features, num_bins, num_planes):
        self.num_bins = num_bins
        self.num_planes = num_planes
        assert num_planes <= num_features, "Number of planes must be <= number of features"
        self.planes = self.generate_orthogonal_planes(num_features, num_planes)

    def generate_orthogonal_planes(self, num_features, num_planes):
        # Generate a random matrix and perform QR decomposition to get orthogonal vectors
        random_matrix = np.random.randn(num_features, num_planes)
        q, _ = np.linalg.qr(random_matrix)
        return q.T  # Transpose to make each row a plane vector

    def hash_vector(self, vector):
        hash_value = 0
        for plane in self.planes:
            if np.dot(vector, plane) >= 0:
                hash_value = (hash_value << 1) | 1
            else:
                hash_value = hash_value << 1
        return hash_value % self.num_bins

    def feature_hashing(self, vectors):
        bins = {i: [] for i in range(self.num_bins)}
        for vector in vectors:
            bin_index = self.hash_vector(vector)
            bins[bin_index].append(vector)
        return bins

def clean_text(text):
    # Remove emojis: Emojis are typically beyond the Basic Multilingual Plane (BMP) of Unicode
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Retain only English alphabets and basic punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def embed(sentence):
    """Returns the embedding for a single sentence."""
    embeddings = model([sentence])
    # Normalize the embedding to use cosine similarity
    return normalize(embeddings)[0]
# def embed(sentence):
#     return model.encode(sentence)

# def kmn(embedding):
#      kmeans.partial_fit(embedding)
#      label = kmeans.predict(embedding)
#      return label[0]
def hsh(vector):
    return lsh.hash_vector(vector)
# df["hasv"]=df["embedding"].apply(hsh)
# result = df[df["hasv"] == 19]["Comment"]
# Load CSV file
df = pd.read_csv('comment/export-youtube-comments.csv')

# # Apply the cleaning function to the Comment column

df['Comment'] = df['Comment'].apply(clean_text)
df["embedding"]=df["Comment"].apply(embed)
#df["label"]=df["embedding"].apply(kmn)
# Group by 'label' and extract sentences
#grouped = df.groupby('label')['Comment']
num_features = len(df["embedding"][0])  
num_vectors = len(df)  
num_bins = 51      
num_planes = 51   
lsh = LSH(num_features, num_bins, num_planes)
df["hasv"]=df["embedding"].apply(hsh)
grouped = df.groupby('hasv')['Comment']
# Print sentences for each label group
for label, sentences in grouped:
    print(f"Label {label}:")
    for sentence in sentences:
        print(f" - {sentence}")
    print()  # For better separation of groups





# print(result)
# print(df.groupby("hasv").count())




    
# Define the function to check for improvement suggestion
def check_improvement(state):
    statement = state['messages'][-1]['statement']
    context= state['messages'][-1]['context']
    response = llm.invoke(
        "Based on the following context of a video:"+context+",does the following comment suggest improvement of any particular sort, reply with simple, without any explanation, yes or no?: " 
        + statement
    )
    state['messages'][-1]["response"]= response
    return state
def summarize(state):
    context= state['messages'][-1]['context']
    response=llm.invoke("summarize the following video transcript in around 100 words,start by The video is about...:"+context)
    state['messages'][-1]['context']=response
    return state
# Define where_to_go function to determine the next state
def where_to_go(state):
    print("State",state)
    response = state['messages'][-1]['response']
    if "yes" in response.lower():
        return "continue"
    else:
        return "end"

# Define the function for preprocessing state
def print_out(state):
    print(state['messages'][-1])
def preprocess(state):
    comment=state['messages'][-1]['statement']
    context=state['messages'][-1]['context']
    response=llm.invoke("This is a summary about a video,:"+context+".This is a comment about improvement on the video content:,"+comment+".Concisely,summarize the improvement in less than 20 word. Start with the person suggests..")
    state['messages'].append({"prepocess": response})
    return state

# Create the workflow
workflow = Graph()

# Add nodes to the workflow
workflow.add_node("check_improvement", check_improvement)
workflow.add_node("preprocess", preprocess)
workflow.add_node("print_out", print_out)
workflow.add_node("summarize",summarize)
# Add conditional edges
workflow.add_conditional_edges(
   "check_improvement", 
   where_to_go,
   {
       "continue": "preprocess",
       "end": END
   }
)


workflow.add_edge("preprocess", "print_out")
workflow.add_edge("summarize","check_improvement")
# Set the entry point
#workflow.set_entry_point("check_improvement")
#workflow.set_finish_point("print_out")
workflow.set_entry_point("summarize")
workflow.set_finish_point("print_out")
# Compile the workflow
app = workflow.compile()

#with open('transcript.txt', 'r') as file:
#    file_contents = file.read()
# Test the workflow
#input_data = {"messages": [{"statement": "The video should have demonstrate how AI can help in music","context":file_contents}]}
#app.invoke(input_data)