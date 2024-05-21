from langchain_community.llms import Ollama
from langgraph.graph import END, Graph
import pandas as pd
import re
#from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow_hub as hub
from sklearn.preprocessing import normalize
import copy
llm = Ollama(model="phi3")
#model = SentenceTransformer('all-MiniLM-L6-v2')
# model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model = hub.load(model_url)

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
# def embed(sentence):
#     """Returns the embedding for a single sentence."""
#     embeddings = model([sentence])
#     # Normalize the embedding to use cosine similarity
#     return normalize(embeddings)[0]
def embed(sentence):
    return model.encode(sentence)


def hsh(vector):
    return lsh.hash_vector(vector)

# Define the function to check for improvement suggestion

# def point_made(state):
#     statement = state['messages'][-1]['statement']
#     context= state['messages'][-1]['context']
#     response = llm.invoke(
#         '''Given the summary transcript of a video:'''+context+'''
#         .The following is a comment on such a video:'''
#         + statement+'''
#         .Concisely write about the subject the person is talking about in the comment in around 10 words in single sentence. Sample kind of output example that you should generate
#         :
#         The person talks about what he likes, The person talks about content of the video, The person talks about where he lives...e.t.c .
#         It sould start with The person talks about... followed by the subject'''
#     )
#     state['messages'][-1]["response"]= response
#     return state

def point_made(state):
    statement = state['messages'][-1]['statement']
    context= state['messages'][-1]['context']
    response = llm.invoke(
        '''Given the summary transcript of a video:'''+context+'''
        .The following is a comment on such a video:'''
        + statement+'''
        .Concisely write about the subject the person is talking about in the comment in around 10 words in single sentence. Sample kind of output example that you should generate
        :
        The person dislikes.. , The person suggests... , The person went..., The person thinks..., The person was motivated..., The person jokes... e.t.c .
        The output should start with The person .. followed by maybe suitable verb ... followed by the subject.The comment might have sarcastical tone, interpret it accordingly.'''
    )
    state['messages'][-1]["response"]= response
    return state
def question_asked(state):
    res=state['messages'][-1]["response"]
    ques=llm.invoke('''The following is a answer to a question:'''+res+
                    '''.Concisely derive a single likely question from the above answer. Keep the output in around 5 words in a single sentence.
                    Sample kind of output example question: What does the person(subject) suggest(action verb)... ,Do the person(subject) believe(action verb)..., When did the person(subject) arrive(action verb)... e.t.c.
                    
                    .The question generated should contain at least: question word (Who, when, what, where,why, how...e.t.c) , subject(the person),  and action verb  ''')
    state['messages'][-1]["question"]=ques
    return state
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
    return state['messages'][-1]["response"],state['messages'][-1]["question"]
def preprocess(state):
    comment=state['messages'][-1]['statement']
    context=state['messages'][-1]['context']
    response=llm.invoke("This is a summary about a video,:"+context+".This is a comment about improvement on the video content:,"+comment+".Concisely,summarize the improvement in less than 20 word. Start with the person suggests..")
    state['messages'].append({"prepocess": response})
    return state

# Create the workflow
workflow = Graph()
workflow2=Graph()
# Add nodes to the workflow
#workflow.add_node("check_improvement", check_improvement)
#workflow.add_node("preprocess", preprocess)
workflow2.add_node("print_out", print_out)
workflow.add_node("summarize",summarize)
workflow2.add_node("point_made",point_made)
workflow2.add_node("question_asked",question_asked)
# Add conditional edges
# workflow.add_conditional_edges(
#    "check_improvement", 
#    where_to_go,
#    {
#        "continue": "preprocess",
#        "end": END
#    }
# )


#workflow.add_edge("preprocess", "print_out")
#workflow.add_edge("summarize","check_improvement")
#workflow.add_edge("summarize","point_made")
workflow2.add_edge("point_made","question_asked")
workflow2.add_edge("question_asked","print_out")
workflow.set_entry_point("summarize")
workflow.set_finish_point("summarize")

workflow2.set_entry_point("point_made")
workflow2.set_finish_point("print_out")
# Compile the workflow
app = workflow.compile()

app2=workflow2.compile()

with open('transcript.txt', 'r') as file:
   file_contents = file.read()

input_data = {"messages": [{"context":file_contents}]}
summary=app.invoke(input_data)
print(summary['messages'][-1]["context"])

def gen_point(comment):
    state=copy.deepcopy(summary)
    state['messages'][-1]['statement']=comment
    point,ques = app2.invoke(state)
    return point,ques
    


df = pd.read_csv('comment/export-youtube-comments.csv')
df=df[:40]
df["Comment"]=df["Comment"].apply(clean_text)
df[['point_made', 'ques_made']] = df['Comment'].apply(lambda x: pd.Series(gen_point(x)))
df.to_csv('point_made/point_made8.csv', index=False)

# df["hasv"]=df["embedding"].apply(hsh)
# grouped = df.groupby('hasv')['Comment']


# # Print sentences for each label group
# for label, sentences in grouped:
#     print(f"Label {label}:")
#     for sentence in sentences:
#         print(f" - {sentence}")
#     print()  # For better separation of groups