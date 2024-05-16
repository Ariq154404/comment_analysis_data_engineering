
from langchain_community.llms import Ollama
from typing import TypedDict, Annotated
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage
llm = Ollama(model="phi3")
class AgentState(TypedDict):
    input_text: str
    is_sarcastic: bool
    important_info: str

# Initialize the StateGraph
state = {
    'input_text': "",
    'is_sarcastic': False,
    'important_info': ""
}
print(llm.invoke("does the following statement suggest  improvement of any particular sort ,reply with simple , without any explanation, yes or no?: Everyone likes working on weekend, don't they? "))