from langchain_community.llms import Ollama
from langgraph.graph import END, Graph

llm = Ollama(model="phi3")


# Define the function to check for improvement suggestion
def check_improvement(state):
    statement = state['messages'][-1]['statement']
    response = llm.invoke(
        "does the following statement suggest improvement of any particular sort, reply with simple, without any explanation, yes or no?: " 
        + statement
    )
    state['messages'].append({"statement": response})
    return state

# Define where_to_go function to determine the next state
def where_to_go(state):
    statement = state['messages'][-1]['statement']
    if "yes" in statement.lower():
        return "continue"
    else:
        return "end"

# Define the function for preprocessing state
def print_out(state):
    print(state['messages'][-1]['statement'])
def preprocess(state):
    state['messages'].append({"statement": "Preprocess state reached"})
    return state

# Create the workflow
workflow = Graph()

# Add nodes to the workflow
workflow.add_node("check_improvement", check_improvement)
workflow.add_node("preprocess", preprocess)
workflow.add_node("print_out", print_out)

# Add conditional edges
workflow.add_conditional_edges(
    "check_improvement", 
    where_to_go,
    {
        "continue": "preprocess",
        "end": END
    }
)

# Add normal edge from `preprocess` to `check_improvement`
workflow.add_edge("preprocess", "print_out")

# Set the entry point
workflow.set_entry_point("check_improvement")
workflow.set_finish_point("print_out")
# Compile the workflow
app = workflow.compile()

# Test the workflow
input_data = {"messages": [{"statement": "Everyone likes working on weekend, don't they?"}]}
app.invoke(input_data)