import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import settings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

async def get_data_from_gpt(system_template, human_template, input) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_template),
        ]
    )
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=1, 
        api_key=settings.OPENAI_API_KEY, 
        max_tokens=1000
    )

    chain = prompt | llm
    response = chain.invoke(input)
    return response.content