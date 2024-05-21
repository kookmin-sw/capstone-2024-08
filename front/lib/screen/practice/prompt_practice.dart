import 'dart:async';
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/practice/prompt_result.dart';
import 'package:capstone/widget/practice/prompt_recording_section.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;

class PromptPractice extends StatefulWidget {
  PromptPractice(
      {super.key,
      required this.script,
      required this.scriptType,
      this.guideVoicePath,
      this.record});

  final ScriptModel script;
  final String scriptType;
  final String? guideVoicePath;
  final RecordModel? record;

  @override
  State<PromptPractice> createState() => _PromptPracticeState();
}

class _PromptPracticeState extends State<PromptPractice> {
  final ScrollController _scrollController = ScrollController();
  String? practiceVoicePath;
  bool showPlayer = false;

  @override
  void initState() {
    super.initState();

    if (widget.guideVoicePath == null) {}

    // 녹음 시작
    Timer.periodic(Duration(milliseconds: 500), (Timer timer) {
      // 스크롤이 더 내려갈 수 있는지 확인
      if (_scrollController.hasClients) {
        // 한 픽셀씩 아래로 스크롤
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(seconds: 3),
          curve: Curves.easeIn,
        );
      }
    });
  }

  nextButtonPressed() async {
    setState(() {
      showPlayer = false;
    });

    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(
          builder: (context) => PromptResult(
              script: widget.script,
              scriptType: widget.scriptType,
              guideVoicePath: widget.guideVoicePath,
              practiceVoicePath: practiceVoicePath)),
      (route) => false, // 모든 이전 화면을 스택에서 제거
    );
  }

  Widget nextButton() {
    return Container(
        margin: const EdgeInsets.all(10),
        child: ElevatedButton(
          onPressed: () {
            debugPrint('녹음 완료 상태 : $showPlayer');
            if (showPlayer) {
              nextButtonPressed();
            } else {
              showDialog(
                context: context,
                builder: (BuildContext context) {
                  return AlertDialog(
                    title: const Text(
                      '잠시만요!',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    content: Text(texts.warningMessage['getUserVoice']!),
                    actions: <Widget>[
                      TextButton(
                        onPressed: () {
                          Navigator.of(context).pop();
                        },
                        child: Text(texts.okButtonText),
                      ),
                    ],
                  );
                },
              );
            }
          },
          style: ButtonStyle(
              elevation: MaterialStateProperty.all<double>(5),
              shape: MaterialStateProperty.all<OutlinedBorder>(
                RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
              backgroundColor: MaterialStateProperty.all<Color>(
                  colors.buttonColor)), // 값을 변경하도록 수정
          child: Text(
            '완료',
            style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: colors.themeWhiteColor),
          ), // 버튼 텍스트 추가
        ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: colors.textColor,
        body: Stack(children: [
          Container(
              padding: EdgeInsets.fromLTRB(20, 40, 20, 60),
              child: ListView.builder(
                controller: _scrollController,
                itemCount: widget.script.content.length, // 텍스트 아이템의 개수
                itemBuilder: (BuildContext context, int index) {
                  // 텍스트 아이템 생성
                  return ListTile(
                    title: Text(
                      widget.script.content[index],
                      style: TextStyle(
                          color: colors.themeWhiteColor, fontSize: 40),
                    ),
                  );
                },
              )),
          Container(
              alignment: Alignment.bottomCenter,
              padding: EdgeInsets.fromLTRB(20, 5, 20, 5),
              child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
                Container(
                    child: PromptRecordingSection(
                  showPlayer: showPlayer,
                  audioPath: '',
                  onDone: (bool isShowPlayer, String? path) {
                    setState(() {
                      showPlayer = isShowPlayer;
                      practiceVoicePath = path;
                    });
                  },
                )),
                Container(
                    padding: const EdgeInsets.all(10),
                    child: !showPlayer ? Container() : nextButton())
              ]))
        ]));
  }
}
