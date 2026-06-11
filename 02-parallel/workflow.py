"""
This workflow having three parallel llm calls, 
these llm calls will generate three different tone of input prompt/input sentence
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
import os
load_dotenv()



# LLM Model
llm = ChatOpenAI(base_url=os.getenv("BASE_URL"),model=os.getenv("MODEL"), api_key=os.getenv("API_KEY")) 


# Define State
class AgentState(TypedDict):
    sentence : str
    formal_tone : str
    casual_tone : str
    friendly_tone: str
    summary: str

def gen_formal_tone(state: AgentState):
    prompt = f"Rewrite the following sentence in formal tone ONLY without changing the meaning: {state['sentence']}"
    formal_tone = llm.invoke(prompt).content

    return {'formal_tone': formal_tone}

def gen_casual_tone(state: AgentState):
    prompt = f"Rewrite the following sentence in casual tone ONLY without changing the meaning: {state['sentence']}"
    casual_tone = llm.invoke(prompt).content

    return {'casual_tone': casual_tone}

def gen_friendly_tone(state: AgentState):
    prompt = f"Rewrite the following sentence in friendly tone ONLY without changing the meaning: {state['sentence']}"
    friendly_tone = llm.invoke(prompt).content

    return {'friendly_tone': friendly_tone}

def summary_of_tone(state: AgentState):
    state['summary'] = f"""
                    The sentence in different tones: \n
                    Original sentence:  {state['sentence']} \n
                    Formal sentence: {state['formal_tone']} \n
                    Casual sentence: {state['casual_tone']} \n
                    Friendly sentence: {state['friendly_tone']}
    """
    return state

# parallel tone
graph = StateGraph(AgentState)
graph.add_node('formal_tone_node', gen_formal_tone)
graph.add_node('casual_tone_node', gen_casual_tone)
graph.add_node('friendly_tone_node', gen_friendly_tone)
graph.add_node('summary_of_tone', summary_of_tone)

graph.add_edge(START, 'formal_tone_node')
graph.add_edge(START, 'casual_tone_node')
graph.add_edge(START, 'friendly_tone_node')

graph.add_edge('formal_tone_node', 'summary_of_tone')
graph.add_edge('casual_tone_node', 'summary_of_tone')
graph.add_edge('friendly_tone_node', 'summary_of_tone')

graph.add_edge('summary_of_tone', END)

workflow = graph.compile()
mystate = workflow.invoke({'sentence': 'You let me doing it you know, i will update you once i complete this work'})
print(mystate)