from create_script.schemas.gpt_sch import GptRequestSch
from create_script.config.config import settings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

async def create_script_by_gpt(req: GptRequestSch) :
    system_template = """
        아나운서가 뉴스에서 읽을 대본을 작성해줘.
        사람들에게 구체적인 정보를 전달하는 것을 목적으로 뉴스 대본을 8줄 작성해줘.
        인사말은 빼고, 제목과 카테고리 언급 없이 뉴스에서 전달할 내용만 만들어줘.
    """

    human_input = "{title} 분야의 {category}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_input),
        ]
    )
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=1, 
        api_key=settings.OPENAI_API_KEY, 
        max_tokens=1000
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "title": req.title,
            "category": req.category
        }
    )

    return response.content