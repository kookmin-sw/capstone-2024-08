import 'package:capstone/model/save_data.dart';
import 'package:flutter/material.dart';

final SaveData saveData = SaveData();

IconButton scrapsButton(String scriptType, String scriptId, String uid,
    int sentenceIndex, bool isClicked) {
  return isClicked
      ? IconButton(
          icon: const Icon(Icons.bookmark),
          onPressed: () {
            saveData.cancelScrap(scriptType, scriptId, uid, sentenceIndex);
          })
      : IconButton(
          icon: const Icon(Icons.bookmark_outline),
          onPressed: () {
            saveData.scrap(scriptType, scriptId, uid, sentenceIndex);
          });
}
