from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from typing import TypedDict
import os
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    model=os.getenv("MODEL")
)

class SubState(TypedDict):
    answer: str
    hindi_translate: str

def hindi_translate_node(state: SubState):
    response = llm.invoke(f"You are a helpful translator, translate English sentence/paragraph into Hindi. Note DO NOT change the meaning of sentence/paragraph.: {state["answer"]}")
    return {"hindi_translate": response.content}

subgraph = StateGraph(SubState)
subgraph.add_node("hindi_translate", hindi_translate_node)
subgraph.add_edge(START, "hindi_translate")
subgraph_node = subgraph.compile()


class ParentState(TypedDict):
    user_sentence: str
    answer: str
    hindi_translate: str

def answer_node(state: ParentState):
        
    response = llm.invoke(f"""You are a helpful chatbot, answer user's query in very polite and consicly: {state["user_sentence"]}""")

    return {"answer": response.content}
def call_subgraph(state: ParentState):
    response_subgraph = subgraph_node.invoke({"answer": state["answer"]})
    return {"hindi_translate": response_subgraph["hindi_translate"]}

myconfig = {"configurable": {"thread_id": "1"}}
checkpointer = InMemorySaver()

parentgraph = StateGraph(ParentState)
parentgraph.add_node("answer_node", answer_node)
parentgraph.add_node("call_subgraph", call_subgraph)
parentgraph.add_edge(START, "answer_node")
parentgraph.add_edge("answer_node", "call_subgraph")
parentgraph.add_edge("call_subgraph", END)

agent  = parentgraph.compile(checkpointer=checkpointer)
response = agent.invoke({"user_sentence": "What is Tool Calling?"}, config=myconfig)
print(response)