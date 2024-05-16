import 'package:capstone/model/record.dart';
import 'package:capstone/model/save_data.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/screen/practice/one_sentence_practice.dart';
import 'package:capstone/screen/practice/prompt_practice.dart';
import 'package:capstone/widget/practice/prompt_timer.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:flutter/services.dart';
import 'package:get/get.dart';

class SelectPractice extends StatelessWidget {
  SelectPractice({
    required this.script,
    required this.tapCloseButton,
    required this.scriptType,
    this.record,
    super.key,
  });

  final ScriptModel script;
  final Function tapCloseButton;
  final SaveData saveData = SaveData();
  String uid = Get.find<UserController>().userModel.id!;

  final String scriptType;
  RecordModel? record;

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);
    var deviceHeight = getDeviceHeight(context);

    return Scaffold(
        body: Container(
            width: deviceWidth,
            height: deviceHeight,
            color: colors.selectPracticebgrColor,
            child: Column(children: [
              Align(
                  alignment: Alignment.topRight,
                  child: IconButton(
                    icon: const Icon(Icons.close_rounded,
                        color: colors.blockColor, size: 40),
                    onPressed: () {
                      tapCloseButton();
                    },
                  )),
              SizedBox(height: deviceHeight * 0.25),
              Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    notice(),
                    SizedBox(height: deviceHeight * 0.05),
                    practiceButton(context, '프롬프트', () {
                      saveData.updateLastPracticeScript(
                          uid, scriptType, script.id!);
                      Get.to(() => PromptTimer(
                          script: script,
                          scriptType: scriptType,
                          record: record,
                          route: 'play_guide'));
                      // PromptPractice(
                      //       script: script,
                      //     ));
                    }),
                    SizedBox(height: deviceHeight * 0.05),
                    practiceButton(context, '문장단위연습', () {
                      saveData.updateLastPracticeScript(
                          uid, scriptType, script.id!);
                      Get.to(() => OneSentencePratice(
                            script: script,
                            scriptType: scriptType,
                            record: record,
                          ));
                    })
                  ])
            ])));
  }
}

Text notice() {
  return const Text('연습 방법을 선택해주세요.',
      style: TextStyle(
        color: colors.blockColor,
        fontSize: 13,
        fontWeight: FontWeight.w500,
      ));
}

Container practiceButton(
    BuildContext context, String buttonText, Function pressedFunc) {
  return Container(
      width: getDeviceWidth(context) * 0.8,
      height: getDeviceHeight(context) * 0.08,
      child: ElevatedButton(
          style: ElevatedButton.styleFrom(
              backgroundColor: colors.blockColor,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15),
              )),
          onPressed: () {
            HapticFeedback.lightImpact();
            pressedFunc();
          },
          child: Text(
            buttonText,
            semanticsLabel: buttonText,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: colors.textColor,
              fontSize: 13,
              fontWeight: FontWeight.w800,
            ),
          )));
}
