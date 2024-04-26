from Levenshtein import distance


def dist(origin, comp):
    # 레벤슈타인 거리 계산
    edit_distance = distance(origin, comp)
    # print(f"Levenshtein Distance: {edit_distance}")
    origin_ipa_len = len(origin)
    modified_ipa_len = len(comp)
    total_length = (origin_ipa_len + modified_ipa_len) / 2
    similarity_percentage = (1 - (edit_distance / total_length)) * 100
    print(f"Similarity Percentage: {similarity_percentage}%")
    return similarity_percentage