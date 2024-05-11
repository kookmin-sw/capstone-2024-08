def modify_category(category):
    if category == 'IT_과학':
        return 'IT/과학'
    elif category == '국제':
        return '세계'
    elif category in ['문화', '스포츠', '지역']:
        return '생활/문화'
    elif category in ['정치', '경제', '사회']:
        return category
