from langfuse import get_client
from langfuse.langchain import CallbackHandler


from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import List, Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import os, sqlite3

load_dotenv()

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """This perform basic +,-,x and /"""
    if operation == '-':
        result = first_num - second_num
    elif operation == '+':
        result = first_num + second_num
    elif operation == '*':
        result = first_num * second_num
    elif operation == '/':
        result = first_num / second_num
    
    return {"result" : result}
all_tools = [search_tool,calculator]
llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
).bind_tools(tools=all_tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Chatbot State
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: ChatState):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

# tool node
tool_node = ToolNode(all_tools)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chatbot_node", chatbot)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "chatbot_node")
graph.add_conditional_edges("chatbot_node", tools_condition, {"tools": "tool_node", "__end__": END})
graph.add_edge("tool_node", "chatbot_node")

config={"configurable": {"thread_id": "1"}, "callbacks": [langfuse_handler]}

workflow = graph.compile(checkpointer=checkpointer)

while True:
    usermessage = input("You: ")
    if usermessage in ['quit', 'bye', 'exit']:
        break
    response = workflow.invoke({'messages': [HumanMessage(content=usermessage)]}, config=config)
    print('\nChatBot: ', response['messages'][-1].content)
