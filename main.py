from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
load_dotenv()

def run_model(llm, prompt_template, information):
    chain = prompt_template | llm
    return chain.invoke(input = {"information": information})


def main():
    print("Hello from langchain-course!")
    information = """
    Brahmagupta (c. 598 - c. 668 CE) was an Indian mathematician and astronomer. He is the author of two early works on mathematics and astronomy: the Brāhmasphuṭasiddhānta (BSS, "correctly established doctrine of Brahma", dated 628), a theoretical treatise, and the Khandakhadyaka ("edible bite", dated 665), a more practical text.
    In 628 CE, Brahmagupta first described gravity as an attractive force, and used the term "gurutvākarṣaṇam" in Sanskrit to describe it.[1][2][3][4] He is also credited with the first clear description of the quadratic formula (the solution of the quadratic equation)[5] in his main work, the Brāhma-sphuṭa-siddhānta.[6]
    """
    summary_template = """
    Given information about a person {information} I want you to create:
    1. A short summary within 100 words
    2. Two interesting facts about them
    """

    prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    llm_openai = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm_ollama = ChatOllama(temperature=0, model="gemma3:270m")

    response_openai = run_model(llm_openai, prompt_template, information)
    print(response_openai.content)
    
    response_ollama = run_model(llm_ollama, prompt_template, information)
    print(response_ollama.content)
    

if __name__ == "__main__":
    main()