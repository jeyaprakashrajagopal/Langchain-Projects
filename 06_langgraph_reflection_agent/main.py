from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from chains import generation_chain, reflection_chain

load_dotenv()

class MessagesGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def should_continue(state: MessagesGraph) -> str:
    if len(state["messages"]) > 6:
        return END
    return REFLECT

REFLECT = "reflect"
GENERATE = "generate"
LAST = -1

def generation_node(state: MessagesGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessagesGraph):
    result = reflection_chain.invoke({"messages": state["messages"]})

    return {"messages": [HumanMessage(content=result.content)]}

flow = StateGraph(MessagesGraph)
flow.add_node(GENERATE, generation_node)
flow.add_node(REFLECT, reflection_node)

flow.set_entry_point(GENERATE)

flow.add_conditional_edges(GENERATE, should_continue, path_map={
    END:END,
    REFLECT:REFLECT
})

flow.add_edge(REFLECT, GENERATE)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="reflection.png")

if __name__ == "__main__":
    print("Reflection agent")
    result = app.invoke({"messages": [HumanMessage(
        content="""Make this tweet better:" 
        @LangChainAI
        — newly Tool Calling feature is seriously underrated.
        After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.
        Made a video covering their newest blog post
        """
    )]})
    print(result)