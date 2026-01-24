from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

@tool
def tripleANumber(number: int) -> int:
    """    
    param number: a number to triple
    return: the triple of the input number
    """
    return number * 3

tools = [TavilySearch(max_results=1), tripleANumber]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools=tools)