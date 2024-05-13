import Levenshtein as lev


def dist(origin, comp):
    # 레벤슈타인 거리 계산
    edit_distance = lev.distance(origin, comp)
    origin_len = len(origin)
    comp_len = len(comp)
    total_length = (origin_len + comp_len) / 2

    # 유사도가 음수가 되는 경우를 방지
    if edit_distance > total_length:
        similarity_percentage = 0
    else:
        similarity_percentage = (1 - (edit_distance / total_length)) * 100

    print(f"Similarity Percentage: {similarity_percentage}%")
    return similarity_percentage