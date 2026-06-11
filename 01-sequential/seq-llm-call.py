"""
This workflow having two llm calls, first will generate topic of blog given question by user,
and second llm call generate the the full blog on that topic.
"""
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
import os

# load .env variables
load_dotenv()

# Define llm
llm = ChatOpenAI(base_url=os.getenv("BASE_URL"),model=os.getenv("MODEL"), api_key=os.getenv("API_KEY")) 

# Define State
class AgentState(TypedDict):
    topic_name: str
    outline: str
    blog_content: str

# Define the node of the workflow for topic of blog
def gen_outline(state: AgentState):
    topic = state["topic_name"]
    prompt = f" Write a detailed outline for blog on following topic: {topic}"
    outline = llm.invoke(prompt).content
    state['outline'] = outline
    return state

def gen_blog(state: AgentState):
    prompt = f" Write a detailed blog on following outlines: {state["outline"]}"
    state["blog_content"] = llm.invoke(prompt).content

    return state

graph = StateGraph(AgentState)

graph.add_node('gen_outline', gen_outline)
graph.add_node('gen_blog', gen_blog)
graph.add_edge(START, 'gen_outline')
graph.add_edge('gen_outline', 'gen_blog')
graph.add_edge('gen_blog', END)

workflow = graph.compile()


print(workflow.invoke({'topic_name': 'Traditional ML in 2026'}))