import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from gpt import get_data_from_gpt

async def modify_content_by_gpt(input):
    system_template = """
        아나운서가 뉴스에서 전달할 내용으로 여섯 문장 정도의 대본을 만들어. 
        인사말 빼고, 뉴스라는 특성에 맞게 아나운서 말투(높임말)로 수정해.
        문장간 내용이 자연스럽게 이어지는 뉴스 대본이 되는 것에 집중해.
        데이터 전체 다 사용하지 않고, 일부만 사용해도 돼. 한자는 빼줘. 
    """
    human_template = "{input}"
    input = {
                "input": input
            }
    
    response = await get_data_from_gpt(
        system_template=system_template,
        human_template=human_template,
        input=input
    )

    content = [sentence.strip() + '.' for sentence in response.split('.')[:-1]]

    return content