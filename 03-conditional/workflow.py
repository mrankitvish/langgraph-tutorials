from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# LLM
llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
)

# Structured output schema
class SentimentFeedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Classify the feedback sentiment."
    )
    intent: Literal["complain", "suggestion", "praise"] = Field(
        description="Identify the user's intent."
    )


structured_llm = llm.with_structured_output(SentimentFeedback)


# Workflow State
class MyState(TypedDict):
    feedback: str
    sentiment: str
    intent: str
    reply: str


# -----------------------------
# Nodes
# -----------------------------

def feedback_analysis_node(state: MyState) -> MyState:
    result = structured_llm.invoke(state["feedback"])

    state["sentiment"] = result.sentiment
    state["intent"] = result.intent

    return state


def intent_router(state: MyState) -> MyState:
    """Dummy node used only for routing."""
    return state


def support_team_node(state: MyState) -> MyState:
    prompt = f"""
    A customer submitted the following complaint:

    "{state['feedback']}"

    Write a short, empathetic customer support response.
    """

    state["reply"] = llm.invoke(prompt).content
    return state


def suggestion_team_node(state: MyState) -> MyState:
    prompt = f"""
    A customer submitted the following suggestion:

    "{state['feedback']}"

    Write a short response thanking them and acknowledging their suggestion.
    """

    state["reply"] = llm.invoke(prompt).content
    return state


def reply_node(state: MyState) -> MyState:
    prompt = f"""
    A customer submitted positive feedback:

    "{state['feedback']}"

    Write a short thank-you response.
    """

    state["reply"] = llm.invoke(prompt).content
    return state


# -----------------------------
# Routing Functions
# -----------------------------

def route_by_sentiment(
    state: MyState,
) -> Literal["reply_node", "intent_router"]:

    if state["sentiment"] == "positive":
        return "reply_node"

    return "intent_router"


def route_by_intent(
    state: MyState,
) -> Literal["support_team_node", "suggestion_team_node"]:

    if state["intent"] == "complain":
        return "support_team_node"

    return "suggestion_team_node"


# -----------------------------
# Build Graph
# -----------------------------

graph = StateGraph(MyState)

graph.add_node("feedback_analysis_node", feedback_analysis_node)
graph.add_node("intent_router", intent_router)
graph.add_node("support_team_node", support_team_node)
graph.add_node("suggestion_team_node", suggestion_team_node)
graph.add_node("reply_node", reply_node)

graph.add_edge(START, "feedback_analysis_node")

graph.add_conditional_edges(
    "feedback_analysis_node",
    route_by_sentiment,
    {
        "reply_node": "reply_node",
        "intent_router": "intent_router",
    },
)

graph.add_conditional_edges(
    "intent_router",
    route_by_intent,
    {
        "support_team_node": "support_team_node",
        "suggestion_team_node": "suggestion_team_node",
    },
)

graph.add_edge("support_team_node", END)
graph.add_edge("suggestion_team_node", END)
graph.add_edge("reply_node", END)

workflow = graph.compile()


# -----------------------------
# Test
# -----------------------------

result = workflow.invoke(
    {
        # "feedback": "Your app is a piece of shit! It crashes every time I open it."
        # "feedback": "Your app is overall good, i didnt face any major issues 4 stars baby!"
        "feedback": "it is taking too long to send msg, I suggest using redis to fix it"
    }
)

print("\nResult:")
print(result)