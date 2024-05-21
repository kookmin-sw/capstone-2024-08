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
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/constants/fonts.dart' as fonts;
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
              SizedBox(height: deviceHeight * 0.02),
              Align(
                  alignment: Alignment.topRight,
                  child: IconButton(
                    icon: Icon(Icons.close_rounded,
                        color: colors.blockColor, size: deviceWidth * 0.1),
                    onPressed: () {
                      tapCloseButton();
                    },
                  )),
              SizedBox(height: deviceHeight * 0.25),
              Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    notice(context),
                    SizedBox(height: deviceHeight * 0.05),
                    practiceButton(context, '프롬프트', () {
                      Get.find<UserController>().updateLastPracticeScript(
                          uid, scriptType, script.id!);
                      promptSelectDialog(context, script, scriptType, record);
                    }),
                    SizedBox(height: deviceHeight * 0.05),
                    practiceButton(context, '문장단위연습', () {
                      Get.find<UserController>().updateLastPracticeScript(
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

Text notice(BuildContext context) {
  return Text('연습 방법을 선택해주세요.',
      style: TextStyle(
        color: colors.blockColor,
        fontSize: fonts.plainText(context),
        fontWeight: FontWeight.w500,
      ));
}

Future<dynamic> promptSelectDialog(context, script, scriptType, record) {
  return showDialog(
    context: context,
    builder: (BuildContext context) {
      return AlertDialog(
        title: Text(
          '어떤 걸 원하시나요?',
          style: TextStyle(
            fontSize: fonts.plainText(context), 
            fontWeight: FontWeight.w800
          ),
        ),
        content: Text(texts.promptStartMessage),
        actionsAlignment: MainAxisAlignment.spaceAround,
        actions: <Widget>[
          ElevatedButton(
            onPressed: () {
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                    builder: (context) => PromptTimer(
                        script: script,
                        scriptType: scriptType,
                        record: record,
                        route: 'play_guide')),
              );
            },
            style: ButtonStyle(
              backgroundColor:
                  MaterialStateProperty.all<Color>(colors.buttonColor),
              shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30.0),
                ),
              ),
            ),
            child: Text(texts.goToPromtGuideText,
                style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: colors.themeWhiteColor)),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                    builder: (context) => PromptTimer(
                        script: script,
                        scriptType: scriptType,
                        record: record,
                        route: 'prompt_practice')),
              );
            },
            style: ButtonStyle(
              backgroundColor:
                  MaterialStateProperty.all<Color>(colors.buttonColor),
              shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30.0),
                ),
              ),
            ),
            child: Text(texts.goToPromtPracticeText,
                style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: colors.themeWhiteColor)),
          )
        ],
      );
    },
  );
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
            style: TextStyle(
              color: colors.textColor,
              fontSize: fonts.plainText(context),
              fontWeight: FontWeight.w800,
            ),
          )));
}
