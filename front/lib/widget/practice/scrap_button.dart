import 'package:capstone/model/save_data.dart';
import 'package:flutter/material.dart';

final SaveData saveData = SaveData();

IconButton scrapsButton(String scriptType, String scriptId, String uid,
    int sentenceIndex, bool isClicked, Function(List<int>?) updateScraps) {
  return isClicked
      ? IconButton(
          icon: const Icon(Icons.bookmark),
          onPressed: () async {
            List<int>? scrapSentence = await saveData.cancelScrap(
                scriptType, scriptId, uid, sentenceIndex);
            updateScraps(scrapSentence);
          })
      : IconButton(
          icon: const Icon(Icons.bookmark_outline),
          onPressed: () async {
            List<int>? scrapSentence =
                await saveData.scrap(scriptType, scriptId, uid, sentenceIndex);
            updateScraps(scrapSentence);
          });
}
