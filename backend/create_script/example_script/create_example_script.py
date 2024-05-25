from datetime import datetime
from langchain.document_loaders.csv_loader import CSVLoader
from modify_category import modify_category
from crawling_content import crawling_content
from modify_data_by_gpt import modify_content_by_gpt
from send_to_db import send_to_db
import asyncio

async def create_example_script():
    loader = CSVLoader(file_path="./news_data/한국언론진흥재단_뉴스빅데이터_메타데이터_언론.csv", encoding='EUC-KR')
    news_dataset = loader.load()[0:30]

    for news_data in news_dataset:
        try:
            data = news_data.page_content.split('\n')

            title = data[4].split(': ')[1]
            
            origin_category = data[5].split(': ')[1].split('>')[0]
            category = modify_category(origin_category)
            if category == None: continue
            
            url = data[0].split(': ')[1]
            origin_content = crawling_content(url)
            content = await modify_content_by_gpt(origin_content)

            script_data = {
                "title": title, 
                "category": category, 
                "createdAt": datetime.utcnow(), 
                "content": content
            }

            send_to_db(script_data)

        except Exception as e:
            print(e)
            continue


async def main():
    await create_example_script()

asyncio.run(main())