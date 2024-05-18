import 'dart:async';
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/widget/practice/prompt_guide_player.dart';
import 'package:capstone/widget/practice/prompt_timer.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;

class PromptGuide extends StatefulWidget {
  PromptGuide(
      {super.key,
      required this.script,
      required this.scriptType,
      required this.record,
      this.guideVoicePath});

  final ScriptModel script;
  final String? guideVoicePath;
  final String scriptType;
  final RecordModel? record;

  @override
  State<PromptGuide> createState() => _PromptGuideState();
}

class _PromptGuideState extends State<PromptGuide> {
  final ScrollController _scrollController = ScrollController();
  bool _isPlaying = true;
  @override
  void initState() {
    super.initState();

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

    Timer(Duration(seconds: 3), () {
      promptSelectDialog(context);
    });
  }

  void _playPause() {
    setState(() {
      _isPlaying = !_isPlaying;
    });
  }

  Future<dynamic> promptSelectDialog(context) {
    return showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text(
            '잠시만요!',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          content: Text(texts.promptStartMessage),
          actionsAlignment: MainAxisAlignment.spaceAround,
          actions: <Widget>[
            ElevatedButton(
              onPressed: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(
                      builder: (context) => PromptGuide(
                            script: widget.script,
                            scriptType: widget.scriptType,
                            record: widget.record,
                            guideVoicePath: widget.guideVoicePath,
                          )),
                );
              },
              style: ButtonStyle(
                backgroundColor:
                    MaterialStateProperty.all<Color>(colors.recordButtonColor),
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
                          script: widget.script,
                          scriptType: widget.scriptType,
                          record: widget.record,
                          route: 'prompt_practice')),
                );
              },
              style: ButtonStyle(
                backgroundColor:
                    MaterialStateProperty.all<Color>(colors.recordButtonColor),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: colors.textColor,
        body: Stack(children: [
          ListView.builder(
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
          GuideVoicePlayer(source: widget.guideVoicePath!, onDelete: () {})
        ])
        // floatingActionButton: FloatingActionButton(
        //   onPressed: _playPause,
        //   child: Icon(_isPlaying ? Icons.pause : Icons.play_arrow),
        // ),
        );
  }
}
