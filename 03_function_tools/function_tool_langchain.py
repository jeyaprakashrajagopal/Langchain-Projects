from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain.tools import tool

@tool
def add(a: int, b: int):
    """Adds both a and b and returns the result"""
    return a + b

tools = [TavilySearch(), add]
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

if __name__ == '__main__':
    print("Tool uses langchain agent for function calling!")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a helpful assistant!"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    result = agent_executor.invoke(input={
        "input": "can you please identify top 10 stocks in USA?"
    })

    print(result)