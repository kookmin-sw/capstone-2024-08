import 'package:capstone/model/save_data.dart';
import 'package:flutter/material.dart';

IconButton scrapsButton(String scriptType, String scriptId, String uid,
    int sentenceIndex, bool isClicked, Function(List<int>?) updateScraps) {
  final SaveData saveData = SaveData();

  return isClicked
      ? IconButton(
          icon: const Icon(Icons.bookmark),
          onPressed: () async {
            List<int>? scrapSentence =
                await saveData.cancelScrap(scriptType, scriptId, sentenceIndex);
            print("언스크랩 함수 내에서의 스크랩 리스트: $scrapSentence");
            updateScraps(scrapSentence);
          })
      : IconButton(
          icon: const Icon(Icons.bookmark_outline),
          onPressed: () async {
            List<int>? scrapSentence =
                await saveData.scrap(scriptType, scriptId, sentenceIndex);
            updateScraps(scrapSentence);
            print("스크랩 함수 내에서의 스크랩 리스트: $scrapSentence");
          });
}
