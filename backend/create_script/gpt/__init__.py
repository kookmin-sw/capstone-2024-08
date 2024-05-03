from create_script.schemas.gpt_sch import GptRequestSch, GptResponseSch
from create_script.config.config import settings
import json, requests

async def create_script_by_gpt(req: GptRequestSch) :
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'user',
                      'content': f'{req.category} 분야의 {req.title}을 제목으로 뉴스 대본 5줄 정도 작성해줘.'}],
        'temperature': 1,
        'max_tokens': 150,
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=settings.OPENAI_HEADERS, json=data)
    output = json.loads(response.text)
    result = output["choices"][0]["message"]["content"]

    return result