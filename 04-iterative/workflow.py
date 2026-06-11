from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, Literal, Annotated, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import operator

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    api_key=os.getenv("API_KEY"),
)

# -------------------------
# State
# -------------------------

class MailReplyer(TypedDict):
    mail: str
    reply: str
    feedback: str
    reply_feedback: Literal["approved", "needs_improvement"]
    iteration: int
    max_iteration: int

    reply_history: Annotated[List[str], operator.add]
    feedback_history: Annotated[List[str], operator.add]


# -------------------------
# Structured Outputs
# -------------------------

class StructuredMailReply(BaseModel):
    reply: str = Field(description="Professional email reply")


class StructuredMailReview(BaseModel):
    reply_feedback: Literal["approved", "needs_improvement"]
    feedback: str = Field(
        description="Reason for approval or improvement suggestions"
    )


reply_llm = llm.with_structured_output(StructuredMailReply)
review_llm = llm.with_structured_output(StructuredMailReview)

# -------------------------
# Nodes
# -------------------------

def write_reply(state: MailReplyer):

    messages = [
        SystemMessage(
            content="""
You are mail replyer.
"""
        ),
        HumanMessage(
            content=f"""
Original Email:

{state['mail']}
"""
        ),
    ]

    response = reply_llm.invoke(messages)

    return {
        "reply": response.reply,
        "reply_history": [response.reply],
        "iteration": 1
    }


def mail_checker(state: MailReplyer):

    messages = [
        SystemMessage(
            content="""
You are a senior email reviewer.

Review the email reply.

Approve only if:
- Professional
- Polite
- Concise
- Clear
- Grammar is correct

Otherwise request improvements.
"""
        ),
        HumanMessage(
            content=f"""
Original Email:

{state['mail']}

Generated Reply:

{state['reply']}
"""
        ),
    ]

    response = review_llm.invoke(messages)

    return {
        "reply_feedback": response.reply_feedback,
        "feedback": response.feedback,
        "feedback_history": [response.feedback],
    }


def fix_reply(state: MailReplyer):

    messages = [
        SystemMessage(
            content="""
You are an expert email editor.

Improve the email reply using the reviewer feedback.

Requirements:
- Professional
- Polite
- Concise
- Natural sounding
- Ready to send
"""
        ),
        HumanMessage(
            content=f"""
Original Email:

{state['mail']}

Current Reply:

{state['reply']}

Reviewer Feedback:

{state['feedback']}
"""
        ),
    ]

    response = reply_llm.invoke(messages)

    return {
        "reply": response.reply,
        "reply_history": [response.reply],
        "iteration": state["iteration"] + 1,
    }


# -------------------------
# Router
# -------------------------

def route_mail_checker(state: MailReplyer):

    if (
        state["reply_feedback"] == "approved"
        or state["iteration"] >= state["max_iteration"]
    ):
        return "approved"

    return "needs_improvement"


# -------------------------
# Graph
# -------------------------

graph = StateGraph(MailReplyer)

graph.add_node("write_reply", write_reply)
graph.add_node("mail_checker", mail_checker)
graph.add_node("fix_reply", fix_reply)

graph.add_edge(START, "write_reply")
graph.add_edge("write_reply", "mail_checker")

graph.add_conditional_edges(
    "mail_checker",
    route_mail_checker,
    {
        "approved": END,
        "needs_improvement": "fix_reply",
    },
)

graph.add_edge("fix_reply", "mail_checker")

workflow = graph.compile()

# -------------------------
# Run
# -------------------------

result = workflow.invoke(
    {
        "mail": "want to take 2 days leave",
        "max_iteration": 3,
    }
)

# print(result)



print("\n" + "=" * 80)
print("📧 EMAIL REPLY WORKFLOW RESULT")
print("=" * 80)

print(f"\n📨 Original Mail:\n{result['mail']}")

print(f"\n✉️ Final Reply:\n{result['reply']}")

print(f"\n✅ Final Status: {result['reply_feedback']}")
print(f"🔄 Iterations: {result['iteration']}")

print("\n📝 Reply History")
print("-" * 80)
for idx, reply in enumerate(result.get("reply_history", []), start=1):
    print(f"\nVersion {idx}")
    print(reply)

print("\n🔍 Review History")
print("-" * 80)
for idx, feedback in enumerate(result.get("feedback_history", []), start=1):
    print(f"\nReview {idx}")
    print(feedback)

print("\n" + "=" * 80)