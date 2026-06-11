from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
import os
load_dotenv()

# Define State
class AgentState(TypedDict):
    question : str
    answer : str

# LLM Model
llm = ChatOpenAI(base_url=os.getenv("BASE_URL"),model=os.getenv("MODEL"), api_key=os.getenv("API_KEY")) 

# Node of the graph
def llm_call(state: AgentState) -> AgentState:
    question = state['question']
    prompt = f" Answer the following question: {question}"
    state["answer"] = llm.invoke(prompt).content

    return state

# Add edges
graph = StateGraph(AgentState)
graph.add_node('LLMCall', llm_call)
graph.add_edge(START, 'LLMCall')
graph.add_edge('LLMCall', END)

workflow = graph.compile()

# Invoke the workflow
for chunk in workflow.stream({'question': 'count 1 to 10'}):
    print(chunk)