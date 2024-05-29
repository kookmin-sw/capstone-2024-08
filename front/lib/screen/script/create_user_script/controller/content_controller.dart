import 'package:flutter/material.dart';
import 'package:get/get.dart';

class UserScriptContentController extends GetxController {
  List<TextEditingController>? textEditingControllerList;

  void updateContent(List<String> sentenceList) {
    textEditingControllerList = [
      for (String sentence in sentenceList)
        TextEditingController(text: sentence)
    ];
  }

  void addController() {
    textEditingControllerList!.add(TextEditingController());
    update();
  }

  void removeController(int idx) {
    textEditingControllerList!.removeAt(idx);
    update();
  }

}
