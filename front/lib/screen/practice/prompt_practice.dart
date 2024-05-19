import 'dart:async';
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/practice/prompt_result.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

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
          duration: Duration(seconds: 8),
          curve: Curves.easeIn,
        );
      }
    });

    // .then((_) {
    //       // 5초 후에 PromptResult 페이지로 이동
    //       Timer(Duration(seconds: 5), () {
    //         // 녹음 중단 및 결과 페이지로 넘어가기 (결과 페이지에는 녹음 저장 위치 넘기기)
    //         Navigator.pushAndRemoveUntil(
    //           context,
    //           MaterialPageRoute(
    //               builder: (context) => PromptResult(
    //                     script: widget.script,
    //                     scriptType: widget.scriptType,
    //                     guideVoicePath: widget.guideVoicePath,
    //                   )),
    //           (route) => false, // 모든 이전 화면을 스택에서 제거
    //         );
    //       });
    //     })
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: colors.textColor,
      body: ListView.builder(
        controller: _scrollController,
        itemCount: widget.script.content.length, // 텍스트 아이템의 개수
        itemBuilder: (BuildContext context, int index) {
          // 텍스트 아이템 생성
          return ListTile(
            title: Text(
              widget.script.content[index],
              style: TextStyle(color: colors.themeWhiteColor, fontSize: 40),
            ),
          );
        },
      ),
    );
  }
}
