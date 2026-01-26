from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

reflection_prompt_template = ChatPromptTemplate.from_messages(
    ('system',
     "You are a viral twitter influencer who is going to grade the user's tweet. Generate critique and recommendations for the user tweet.",
     "Always provide the detailled recommendations, including requests for length, virality, style, etc."
     ),
     MessagesPlaceholder(variable_name="messages")
)

generation_prompt_template = ChatPromptTemplate.from_messages(
    ('system',
     "You are a best twitter tweets writter who specializes in creating viral posts.",
     "Generate the best twitter posts possible with user's request."
     "You should provide improved version of previous attempts of the twitter tweet based on the user critique and the recommendations."
     ),
     MessagesPlaceholder(variable_name="messages")
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

reflection_chain = reflection_prompt_template | llm
generation_chain = generation_prompt_template | llm