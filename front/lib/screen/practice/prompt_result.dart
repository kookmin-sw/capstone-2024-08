import 'dart:io';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/save_data.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/model/save_data.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/screen/practice/guide_voice_player.dart';
import 'package:capstone/widget/audio_player.dart';
import 'package:capstone/widget/audio_recoder/recording_section.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class PromptResult extends StatefulWidget {
  PromptResult(
      {super.key,
      required this.script,
      required this.scriptType,
      this.guideVoicePath,
      this.record});

  final ScriptModel script;
  final String scriptType;
  final String? guideVoicePath;
  RecordModel? record;

  @override
  State<PromptResult> createState() => _PromptResultState();
}

class _PromptResultState extends State<PromptResult> {
  final Map<String, File?> _wavFiles = Get.find<UserController>().wavFiles;
  final SaveData saveData = SaveData();
  int? sentenceLength;
  String uid = Get.find<UserController>().userModel.id!;
  String? practiceAudioPath;

  bool showPlayer = false;
  bool showGuideVoicePlayer = false;
  Map<String?, String?>? practiceResult;
  int? _precision;

  late Widget? _guideVoicePlayers;

  @override
  void initState() {
    showPlayer = false;
    showGuideVoicePlayer = false;
    super.initState();
  }

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
          fontSize: 15, fontWeight: FontWeight.w800, color: colors.textColor),
    );
  }

  nextButtonPressed() async {
    setState(() {
      showPlayer = false;
      showGuideVoicePlayer = false;
    });
    saveData.updatePromptPracticeResult(
        scriptId: widget.script.id!,
        scriptType: widget.scriptType,
        precision: _precision);
    backToHomePage();
  }

  backToHomePage() {
    // 연습 기록 저장 코드
    Future.delayed(Duration(milliseconds: 700), () {
      Navigator.pushNamedAndRemoveUntil(
        context,
        '/bottom_nav',
        (Route<dynamic> route) => false,
      );
    });
  }

  Widget sentenceSection() {
    // 이미 호출된 guideVoicePlayer() 함수의 결과를 저장하고 있다가 재사용
    if (_guideVoicePlayers != null) {
      _guideVoicePlayers = FutureBuilder<Widget>(
        future: guideVoicePlayer(widget.script.content.join(' ')),
        builder: (context, snapshot) {
          return waitingGetGuideVoicePlayer(snapshot);
        },
      );
    }
    return Container(
      width: MediaQuery.of(context).size.width / 1.2,
      padding: EdgeInsets.all(20.0),
      decoration: BoxDecoration(
          color: colors.themeWhiteColor,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
                color: Colors.grey.withOpacity(0.5),
                spreadRadius: 2,
                blurRadius: 5,
                offset: Offset(0, 3) // changes position of shadow
                ),
          ]),
      child: Column(children: [
        Container(
          padding: EdgeInsets.fromLTRB(5, 20, 5, 20),
          child: Text(
            widget.script.content.join(' '),
            textAlign: TextAlign.start,
            style: TextStyle(fontSize: 14.0),
          ),
        ),
        _guideVoicePlayers!,
        RecordingSection(
          showPlayer: showPlayer,
          audioPath: '',
          onDone: (bool isShowPlayer, String? path) {
            setState(() {
              showPlayer = isShowPlayer;
              practiceAudioPath = path;
            });
          },
        )
      ]),
    );
  }

  Future<Widget> precisionSection() async {
    if (showPlayer) {
      practiceResult = await getVoicesSimilarity(
          widget.script.content.join(' '), practiceAudioPath!);
      setState(() {
        (practiceResult!['precision'] == null)
            ? _precision = null
            : _precision = int.parse(practiceResult!['precision']!);
      });
      return Column(children: [
        Container(
          child: Text('정확도 : ${_precision}'),
          padding: EdgeInsets.fromLTRB(20, 40, 20, 20),
        ),
        Container(
          child: Text('발음 기호 : ${practiceResult!['pronunciation']}'),
          padding: EdgeInsets.fromLTRB(20, 0, 20, 20),
        ),
      ]);
    }
    return Container();
  }

  Widget nextButton() {
    return Container(
        width: MediaQuery.of(context).size.width / 1.2,
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
            texts.nextButtonText,
            style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: colors.themeWhiteColor),
          ), // 버튼 텍스트 추가
        ));
  }

  Future<Widget> guideVoicePlayer(String text) async {
    return Container(
        padding: EdgeInsets.fromLTRB(20, 0, 20, 0),
        child: (widget.guideVoicePath != null)
            ? AudioPlayer(source: widget.guideVoicePath!, onDelete: () {})
            : Text('가이드 음성 : ${widget.guideVoicePath}'));
  }

  Widget waitingGetGuideVoicePlayer(snapshot) {
    if (snapshot.connectionState == ConnectionState.waiting) {
      // 데이터 로딩 중일 때 표시할 위젯
      return const Column(children: [
        CircularProgressIndicator(
          color: colors.recordButtonColor,
        ),
        SizedBox(height: 20),
        Text(
          '가이드 음성 생성하는 중',
          style: TextStyle(
              color: colors.recordButtonColor, fontWeight: FontWeight.bold),
        )
      ]);
    } else if (snapshot.hasError) {
      // 오류 발생 시 표시할 위젯
      return Text('Error: ${snapshot.error}');
    } else {
      // 데이터를 성공적으로 받아왔을 때 표시할 위젯
      return snapshot.data ?? Container(); // 반환된 위젯을 표시
    }
  }

  Widget waitingGetPrecisionSection(snapshot) {
    if (snapshot.connectionState == ConnectionState.waiting) {
      // 데이터 로딩 중일 때 표시할 위젯
      return Container(
          padding: const EdgeInsets.all(20),
          child: const Column(children: [
            CircularProgressIndicator(
              color: colors.recordButtonColor,
            ),
            SizedBox(height: 20),
            Text(
              '유사도 계산하는 중',
              style: TextStyle(
                  color: colors.recordButtonColor, fontWeight: FontWeight.bold),
            )
          ]));
    } else if (snapshot.hasError) {
      // 오류 발생 시 표시할 위젯
      return Text('Error: ${snapshot.error}');
    } else {
      // 데이터를 성공적으로 받아왔을 때 표시할 위젯
      return snapshot.data ?? Container(); // 반환된 위젯을 표시
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Stack(children: [
      Container(
          padding: const EdgeInsets.fromLTRB(20, 5, 20, 20),
          child: SingleChildScrollView(
              child: Column(children: [
            Column(children: [
              _buildCategory(widget.script.category),
              const SizedBox(height: 15),
              _buildTitle(widget.script.title),
              const SizedBox(height: 20)
            ]),
            Container(
                child: Column(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [sentenceSection()])),
            FutureBuilder<Widget>(
              future:
                  precisionSection(), // precisionSection 함수를 호출하여 Future<Widget>을 얻음
              builder: (context, snapshot) {
                return waitingGetPrecisionSection(snapshot);
              },
            ),
            const SizedBox(
              height: 50,
            )
          ]))),
      Container(
          alignment: Alignment.bottomCenter,
          padding: const EdgeInsets.all(20),
          child: nextButton())
    ]));
  }
}
