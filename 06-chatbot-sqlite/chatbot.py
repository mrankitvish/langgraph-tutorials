from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import List, Annotated, TypedDict
from dotenv import load_dotenv
import os, sqlite3

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
)
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Chatbot State
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: ChatState):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chatbot_node", chatbot)
graph.add_edge(START, "chatbot_node")
graph.add_edge("chatbot_node", END)

config={"configurable": {"thread_id": '1'}}

workflow = graph.compile(checkpointer=checkpointer)

# while True:
#     usermessage = input("You: ")
#     if usermessage in ['quit', 'bye', 'exit']:
#         break
#     response = workflow.invoke({'messages': [HumanMessage(content=usermessage)]}, config=config)
#     print('\nChatBot: ', response['messages'][-1].content)
thread_ids = set()
for i in checkpointer.list(None):
    thread_ids.add(i.config['configurable']['thread_id'])
print(list(thread_ids))