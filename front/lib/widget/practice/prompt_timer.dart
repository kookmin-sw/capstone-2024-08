import 'dart:async';
import 'dart:io';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/screen/practice/guide_voice_player.dart';
import 'package:capstone/screen/practice/prompt_practice.dart';
import 'package:capstone/widget/practice/prompt_guide.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';

class PromptTimer extends StatefulWidget {
  PromptTimer(
      {super.key,
      required this.script,
      required this.scriptType,
      required this.route,
      this.guideVoicePath,
      this.record});

  final ScriptModel script;
  final String scriptType;
  String? guideVoicePath;
  RecordModel? record;
  final String route;

  @override
  State<PromptTimer> createState() => _PromptTimerState();
}

class _PromptTimerState extends State<PromptTimer> {
  Timer? _timer;
  int _second = 3;
  bool _isLandscape = false;

  final Map<String, File?> _wavFiles = Get.find<UserController>().wavFiles;

  bool isPromptGuide() {
    return (widget.route == 'play_guide');
  }

  bool isGuideAudioExist() {
    return (widget.guideVoicePath != null);
  }

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
              builder: (context) => isPromptGuide()
                  ? PromptGuide(
                      script: widget.script,
                      scriptType: widget.scriptType,
                      record: widget.record,
                      guideVoicePath: widget.guideVoicePath)
                  : PromptPractice(
                      script: widget.script,
                      scriptType: widget.scriptType,
                      record: widget.record,
                      guideVoicePath: widget.guideVoicePath),
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

  Future<Widget> afterGetGuidVoiceWidget(String text) async {
    widget.guideVoicePath =
        await sendDataToServerAndDownLoadGuideVoice(text, _wavFiles);
    print("프롬프트에서 생성한 가이드 음성 : ${widget.guideVoicePath}");
    return timerCommonWidget();
  }

  Widget timerCommonWidget() {
    double screenWidth = MediaQuery.of(context).size.width;

    return OrientationBuilder(
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
                color: colors.themeWhiteColor,
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
    );
  }

  Widget waitingGetGuideVoice(snapshot) {
    if (snapshot.connectionState == ConnectionState.waiting) {
      // 데이터 로딩 중일 때 표시할 위젯
      return Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        CircularProgressIndicator(
          color: colors.exampleScriptColor,
        ),
        SizedBox(height: 20),
        Text(
          '가이드 음성 생성하는 중',
          style: TextStyle(
              color: colors.exampleScriptColor, fontWeight: FontWeight.bold),
        )
      ]);
    } else if (snapshot.hasError) {
      // 오류 발생 시 표시할 위젯
      return Text(
        'Error: ${snapshot.error}',
        style: TextStyle(color: colors.themeWhiteColor),
        textAlign: TextAlign.center,
      );
    } else {
      // 데이터를 성공적으로 받아왔을 때 표시할 위젯
      return snapshot.data ?? Container(); // 반환된 위젯을 표시
    }
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;

    return Scaffold(
        backgroundColor: colors.textColor,
        body: Center(
            child: isGuideAudioExist()
                ? FutureBuilder<Widget>(
                    future: afterGetGuidVoiceWidget(
                        widget.script.content.join(' ')),
                    builder: (context, snapshot) {
                      return waitingGetGuideVoice(snapshot);
                    },
                  )
                : timerCommonWidget()));
  }
}
