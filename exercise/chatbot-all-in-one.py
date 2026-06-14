# Imports
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, START,  END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3, os
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# load env
load_dotenv()


# configure RAG
loader = PyPDFLoader("/home/ankit/Projects/langgraph-workflows/08-rag-chatbot/intro-to-ml.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=200,  separators=['\n\n', '\n', ' ', ''])
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = splitter.split_documents(docs)
vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# define tools
web_search = DuckDuckGoSearchRun(region="en-us")

@tool
def rag_tool(query: str):
    """
    Search the PDF knowledge base and return relevant context.
    """
    decision = interrupt(
        {
            "approval": "Approve to retrieve data? yes/no"
        }
    )
    if decision['approval'] == 'yes':
        docs = retriever.invoke(query)
        # print(docs)
        return "\n\n".join(
            [doc.page_content for doc in docs]
        )
    return {"docs": "Denied the access to retrieve data"}

# create LLM
llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    model=os.getenv("MODEL")
).bind_tools([web_search, rag_tool])

# create state
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

# create node for chat
def chat_node(state: ChatState):
    sys_prompt = SystemMessage(content="You are a helpful assistant, reply user's every query in simple, understandable and consicly, you have access to tools 'web_search' for search in web and  'rag_tool' for local RAG PDF document retrieval, use when requires")
    prompt = [sys_prompt] + state["messages"]
    response = llm.invoke(prompt)

    return {"messages": [response]}

# tool node
tool_node = ToolNode([web_search, rag_tool])

# establish the sqlite3 db conn
conn = sqlite3.connect(database="my-test-chatdb.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# set config
config = {"configurable": {"thread_id": "1"}}

# create graph 
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", "__end__": END})
graph.add_edge("tool_node","chat_node")

# compile the graph
chatbot = graph.compile(checkpointer=checkpointer)

# invoke graph

while True:
    user_input = input("Human: ")
    if user_input in ["quit","exit","bye"]:
        break

    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    
    if "__interrupt__" in response:
        ans = input(f"want to approve the request for: {response['__interrupt__'][0].value}")
        final_response = chatbot.invoke(Command(resume={"approval": ans}),config=config)
        print("AI: ", final_response["messages"][-1].content)

    print("AI: ", response["messages"][-1].content)
