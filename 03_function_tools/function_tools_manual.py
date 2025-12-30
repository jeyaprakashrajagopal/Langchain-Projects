from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def fake_weather_api(city: str) -> str:
    """Check the weather in a specified city"""
    return "Cloudy, 25°C"


@tool
def outdoor_seating_availability(city: str) -> str:
    """Check if outdoor activities are possible."""
    return "Outdoor activity is possible!."


tools = [fake_weather_api, outdoor_seating_availability]

llm_with_tools = ChatOpenAI(temperature=0, model="gpt-4o-mini").bind_tools(tools)

agent = llm_with_tools


def main():
    print("Hello from langchain-projects!")

    messages = [
        HumanMessage(
            "How will the weather be in Frankfurt today? I would like to do some outdoor activity."
        )
    ]
    # 1). Invoke LLM to get the tools response
    llm_output = llm_with_tools.invoke(messages)
    # 2). Add AIMessage to the messages
    messages.append(llm_output)

    # 3). Mapping tool names to tools
    tool_mapping = {
        "fake_weather_api": fake_weather_api,
        "outdoor_seating_availability": outdoor_seating_availability,
    }

    # 4). Iterating through tool calls from tool response and append the ToolMessage to messages
    for tool_call in llm_output.tool_calls:
        # Mapping tool name to tool
        tool = tool_mapping[tool_call["name"].lower()]
        # Runs Runnable interface with arguments dictionary
        tool_output = tool.invoke(tool_call["args"])
        # Appends the response as Tool message
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    # Final invocation to get the response based on these tools
    result = llm_with_tools.invoke(messages)

    print(result.content)


if __name__ == "__main__":
    main()
