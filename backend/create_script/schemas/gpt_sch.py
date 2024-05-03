from pydantic import BaseModel

class GptRequestSch(BaseModel):
    title: str
    category: str

    class Config :
        orm_mode = True
        schema_extra = {
            "example": {
                "title": "민간 주도 경제적 우주 개발…‘뉴스페이스’ 시대로",
                "category": "IT/과학"
            }
        }

class GptResponseSch(BaseModel):
    script: str

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "script": "민간이 이끌어가는 우주개발 시대로의 변화를 '뉴 스페이스'라고 하는데요. 어제 발사된 초소형 군집 위성 역시 학계와 민간기업이 주도적으로 참여해 개발에 성공했습니다. 발사의 의미, 지형철 기자가 짚어봤습니다."
            }
        }