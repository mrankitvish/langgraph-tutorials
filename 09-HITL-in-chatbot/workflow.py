from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import os, sqlite3

load_dotenv()


llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
)
# Chatbot State
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: ChatState):

    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a user question.",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no"
    })
    
    if decision["approved"] == 'yes':
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    else:
        return {"messages": [AIMessage(content="Not approved.")]}


# 3. Build the graph: START -> chat -> END
builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", END)

conn = sqlite3.connect(database="hitl-chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Compile the app
app = builder.compile(checkpointer=checkpointer)
# Create a new thread id for this conversation
config = {"configurable": {"thread_id": '1234'}}

# ---- STEP 1: user asks a question ----
initial_input = {
    "messages": [
        ("user", "Explain gradient descent in very simple terms.")
    ]
}

# Invoke the graph for the first time
result = app.invoke(initial_input, config=config)
print(result)

print( result['__interrupt__'][0].value)
user_input = input(f"\nBackend message - {result['__interrupt__'][0].value} \n Approve this question? (yes/no): ")

final_result = app.invoke(
    Command(resume={"approved": user_input}),
    config=config,
)

print(final_result["messages"][-1].content)