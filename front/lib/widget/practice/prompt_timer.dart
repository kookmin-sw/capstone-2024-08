import 'dart:async';
import 'dart:io';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/screen/practice/prompt_practice.dart';
import 'package:capstone/widget/practice/prompt_guide.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class PromptTimer extends StatefulWidget {
  PromptTimer(
      {super.key,
      required this.script,
      required this.scriptType,
      required this.route,
      this.record});

  final ScriptModel script;
  final String scriptType;
  RecordModel? record;
  String? guideVoicePath;
  final String route;

  @override
  State<PromptTimer> createState() => _PromptTimerState();
}

class _PromptTimerState extends State<PromptTimer> {
  Timer? _timer;
  int _second = 3;
  bool _isLandscape = false;

  final Map<String, File?> _wavFiles = Get.find<UserController>().wavFiles;

  // 가이드 음성 다운로드 후 파일 경로 가져오는 코드 작성

  void startTimer() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        if (_second > 0) {
          _second--;
        } else {
          _timer?.cancel(); // 타이머 종료
          // 화면 전환
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (context) => (widget.route == 'play_guide')
                  ? PromptGuide(script: widget.script)
                  : PromptPractice(script: widget.script),
            ),
          );
        }
      });
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;

    return Scaffold(
        backgroundColor: colors.textColor,
        body: Center(child: OrientationBuilder(
          builder: (context, orientation) {
            if (orientation == Orientation.landscape) {
              if (!_isLandscape) {
                _isLandscape = true;
                startTimer();
              }
              return Container(
                  alignment: Alignment.center,
                  width: screenWidth / 4,
                  height: screenWidth / 4,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white,
                  ),
                  child: Text(
                    '$_second',
                    style: TextStyle(
                        color: colors.textColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 48.0),
                    textAlign: TextAlign.center,
                  ));
            } else {
              _isLandscape = false;
              _timer?.cancel();
              _second = 3;
              return Container(
                  padding: EdgeInsets.fromLTRB(20, 0, 20, 0),
                  child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          '${texts.orientationMessage}',
                          style: TextStyle(
                              fontSize: 20, color: colors.themeWhiteColor),
                          textAlign: TextAlign.center,
                        ),
                        Container(
                            padding: EdgeInsets.all(70),
                            child: Icon(
                              Icons.rotate_90_degrees_ccw,
                              size: screenWidth / 4,
                              color: colors.themeWhiteColor,
                            ))
                      ]));
            }
          },
        )));
  }
}
