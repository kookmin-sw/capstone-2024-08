import 'package:flutter/material.dart';
import 'package:get/get.dart';

class UserScriptContentController extends GetxController {
  List<TextEditingController>? textEditingControllerList;
  final List<String> sentenceList;
  UserScriptContentController(this.sentenceList);

  @override
  void onInit() {
    textEditingControllerList = [
      for (String sentence in sentenceList)
        TextEditingController(text: sentence)
    ];
    super.onInit();
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
