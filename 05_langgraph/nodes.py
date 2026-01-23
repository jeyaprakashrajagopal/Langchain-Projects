from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import tools, llm

load_dotenv()


SYSTEM_MESSAGE="""Youre a helpful assistant that can use the tools to answer questions"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Runs the agent reasoning node."""
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)