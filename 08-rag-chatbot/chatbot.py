from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
    encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for cosine similarity
)
# document loader & text splitter
loader = PyPDFLoader('/home/ankit/Projects/langgraph-workflows/08-rag-chatbot/intro-to-ml.pdf')
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=['\n\n', '\n', ' ', ''])
chunks = splitter.split_documents(docs)
len(chunks)
# chuncks into embeddings

vector_store = FAISS.from_documents(chunks, embedding)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# tools
web_search = DuckDuckGoSearchRun(region="us-en")


@tool
def rag_tool(query: str):
    """Retrieve document from vector store by 'query' ."""
    result = retriever.invoke(query)
    len(result)
    return { "retrieved_context": result}
tools = [web_search, rag_tool]
llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
).bind_tools(tools)

class ChatbotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# chat node
def chat_node(state: ChatbotState):
    sys_prompt = SystemMessage(content="""
                You are a helpful chatbot, you have 'rag_tool' access to retrieve relevant context using the tool.
                           you also have tool for websearch 'web_search'.
                           Note: Answer users question in very consice and use the tools when required.

                """)
    prompt = [sys_prompt] + state['messages']
    response = llm.invoke(prompt)
    return {"messages": [response]}

# tool node
conn = sqlite3.connect(database="rag-chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
config={"configurable": {"thread_id": "1"}}

tool_node = ToolNode(tools)

# Build graph
graph = StateGraph(ChatbotState)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {
        "tools": "tool_node",
        "__end__": END
    }
)
graph.add_edge("tool_node", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

while True:
    usermessage = input("\nYou: ")
    if usermessage.lower() in ['quit', 'bye', 'exit']:
        break
        
    # Use stream() with stream_mode="updates" to get real-time node outputs
    events = chatbot.stream(
        {'messages': [HumanMessage(content=usermessage)]}, 
        config=config, 
        stream_mode="updates"
    )
    
    for event in events:
        for node_name, node_state in event.items():
            
            # Intercept messages coming from the LLM
            if node_name == "chat_node":
                last_msg = node_state["messages"][-1]
                
                # 1. Check if the LLM decided to trigger a tool
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    for tool in last_msg.tool_calls:
                        print(f"\n[🛠️  Triggering Tool: {tool['name']}]")
                        print(f"    Arguments: {tool['args']}")
                        
                # 2. If there are no tool calls, this is the final conversational response
                elif last_msg.content:
                    print('\nChatBot:', last_msg.content)
                    
            # Intercept updates coming from your tools
            elif node_name == "tool_node":
                print("[✅ Tool finished. Processing results...]")