import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/save_data.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/script/create_user_script/controller/content_controller.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/bottom_buttons.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:capstone/widget/script/script_content_adjust_block.dart';
import 'package:capstone/screen/script/select_practice.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:capstone/widget/warning_dialog.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class AdjustUserScript extends StatefulWidget {
  const AdjustUserScript({
    Key? key,
    required this.title,
    required this.category,
  }) : super(key: key);

  final String title;
  final String category;

  @override
  State<AdjustUserScript> createState() => _AdjustUserScriptState();
}

class _AdjustUserScriptState extends State<AdjustUserScript> {
  SaveData saveData = SaveData();
  List<String> sentenceList = [];

  Text _buildCategory(String category) {
    return Text(
      category,
      semanticsLabel: category,
      textAlign: TextAlign.start,
      style: const TextStyle(
          fontSize: 12, fontWeight: FontWeight.w500, color: colors.textColor),
    );
  }

  Text _buildTitle(String title) {
    return Text(
      title,
      semanticsLabel: title,
      textAlign: TextAlign.start,
      style: const TextStyle(
          fontSize: 15, fontWeight: FontWeight.w500, color: colors.textColor),
    );
  }

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);
    var deviceHeight = getDeviceHeight(context);

    return Scaffold(
        appBar: basicAppBar(title: '나만의 대본 만들기'),
        body: Stack(children: [
          GestureDetector(
              onTap: () {
                FocusScope.of(context).unfocus();
              },
              child: Container(
                  padding: const EdgeInsets.all(20),
                  child: ListView(children: [
                    _buildCategory(widget.category),
                    SizedBox(height: deviceHeight * 0.01),
                    _buildTitle(widget.title),
                    SizedBox(height: deviceHeight * 0.03),
                    GetBuilder<UserScriptContentController>(
                        builder: (controller) {
                      return scriptContentAdjustBlock(controller, deviceWidth);
                    }),
                    SizedBox(height: deviceHeight * 0.08),
                  ]))),
          bottomButtons(
              deviceWidth,
              outlinedRoundedRectangleButton('저장 후 나가기', () {
                if (checkValidContent()) {
                  saveUserScript();
                  Get.close(2);
                }
              }),
              fullyRoundedRectangleButton(colors.buttonColor, '연습하기', () {
                if (checkValidContent()) {
                  ScriptModel userScript = saveUserScript();
                  RecordModel record = RecordModel(
                      id: userScript.id, scrapSentence: [], promptResult: []);
                  Get.to(() => SelectPractice(
                        script: userScript,
                        scriptType: 'user',
                        tapCloseButton: () {
                          Get.close(3);
                        },
                        record: record,
                      ));
                }
              }))
        ]));
  }

  void showInvalidContentWarning(String warningType) {
    showDialog(
        context: context,
        builder: (BuildContext context) =>
            WarningDialog(warningObject: warningType));
  }

  bool checkValidContent() {
    List<TextEditingController> controllers =
        Get.find<UserScriptContentController>().textEditingControllerList!;
    sentenceList.clear();

    if (controllers.isEmpty) {
      showInvalidContentWarning('emptyContent');
      return false;
    }

    for (TextEditingController controller in controllers) {
      if (controller.text == '') {
        showInvalidContentWarning('emptySentenceBlock');
        return false;
      }
      sentenceList.add(controller.text);
    }
    return true;
  }

  ScriptModel saveUserScript() {
    ScriptModel userScript = ScriptModel(
      title: widget.title,
      category: widget.category,
      content: sentenceList,
      createdAt: Timestamp.now(),
    );

    saveData.addUserScript(userScript);

    Get.delete<UserScriptContentController>();

    return userScript;
  }
}
