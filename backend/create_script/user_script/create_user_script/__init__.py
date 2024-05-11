from ..schemas.gpt_sch import GptRequestSch
from create_script import gpt

async def create_script_by_gpt(req: GptRequestSch) -> str:
    system_template = """
        아나운서가 뉴스에서 읽을 대본을 작성해줘.
        사람들에게 구체적인 정보를 전달하는 것을 목적으로 뉴스 대본을 8줄 작성해줘.
        인사말은 빼고, 제목과 카테고리 언급 없이 뉴스에서 전달할 내용만 만들어줘.
    """
    human_template = "{title} 분야의 {category}"
    input = {
            "title": req.title,
            "category": req.category
            }
    
    script = await gpt.get_data_from_gpt(
        system_template=system_template,
        human_template=human_template,
        input=input
    )

    return script