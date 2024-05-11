import 'package:capstone/widget/audio_recoder/recording_section.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animation_progress_bar/flutter_animation_progress_bar.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';

class GetUserVoice extends StatefulWidget {
  const GetUserVoice({super.key});

  @override
  State<GetUserVoice> createState() => _GetUserVoiceState();
}

class _GetUserVoiceState extends State<GetUserVoice> {
  double _currentProgressValue = 5;
  String _currentState = 'short';
  bool showPlayer = false;
  String? audioPath;

  @override
  void initState() {
    showPlayer = false;
    super.initState();
  }

  getNextState(String currentState) {
    if (currentState == 'short') {
      return 'middle';
    } else if (currentState == 'middle') {
      return 'long';
    } else {
      return 'end';
    }
  }

  nextButtonPressed(String currentState) {
    debugPrint('$showPlayer');

    if (showPlayer) {
      setState(() {
        // wav 파일 저장하는 코드 필요
        _currentState = getNextState(currentState);
        _currentProgressValue = texts.getUserProgressValues[_currentState]!;
        debugPrint("#########################");
        debugPrint(audioPath);
      });
    } else {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text(
              '잠시만요!',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            content: Text(texts.warningMessage['getUserVoice']!),
            actions: <Widget>[
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: Text('확인'),
              ),
            ],
          );
        },
      );
    }
  }

  Widget progressBarSection(double currentValue) {
    return Container(
        padding: const EdgeInsets.fromLTRB(15, 15, 20, 20),
        color: colors.bgrBrightColor,
        child: FAProgressBar(
          currentValue: currentValue,
          maxValue: 100,
          backgroundColor: colors.bgrBrightColor,
          progressColor: colors.bgrDarkColor,
          borderRadius: BorderRadius.circular(20),
          size: 20,
          displayText: '%',
        ));
  }

  Widget subTitleSection() {
    return Container(
        padding: const EdgeInsets.only(top: 30, bottom: 30),
        child: Text(texts.getUserVoiceSubtitle));
  }

  Widget exampleSentenceSection(String exampleSentenceType) {
    return AnimatedSwitcher(
        duration: Duration(milliseconds: 500),
        child: Container(
          width: MediaQuery.of(context).size.width / 1.2,
          key: ValueKey<String>(exampleSentenceType),
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
              child: Text(
                texts.getUserVoiceExampleSentences[exampleSentenceType]!,
                textAlign: TextAlign.start,
                style: TextStyle(fontSize: 14.0),
              ),
              padding: EdgeInsets.fromLTRB(5, 20, 5, 20),
            ),
            RecordingSection(
              showPlayer: showPlayer,
              audioPath: '',
              onDone: (bool isShowPlayer, String? path) {
                setState(() {
                  showPlayer = isShowPlayer;
                  audioPath = path;
                });
              },
            )
          ]),
        ));
  }

  Widget nextButton() {
    return Container(
        width: MediaQuery.of(context).size.width / 1.2,
        margin: const EdgeInsets.all(10),
        child: ElevatedButton(
          onPressed: () => nextButtonPressed(_currentState),
          style: ButtonStyle(
              elevation: MaterialStateProperty.all<double>(5),
              shape: MaterialStateProperty.all<OutlinedBorder>(
                RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
              backgroundColor: MaterialStateProperty.all<Color>(
                  colors.bgrDarkColor)), // 값을 변경하도록 수정
          child: const Text(
            '다음',
            style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: colors.themeWhiteColor),
          ), // 버튼 텍스트 추가
        ));
  }

  AppBar getUserVoiceAppBar() {
    return AppBar(
      backgroundColor: colors.bgrDarkColor,
      centerTitle: true,
      title: Text(texts.getUserVoiceAppBarTitle,
          style: const TextStyle(
              color: colors.themeWhiteColor, fontWeight: FontWeight.bold)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: getUserVoiceAppBar(),
        body: Container(
            color: colors.bgrBrightColor,
            child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(children: [
                    progressBarSection(_currentProgressValue),
                    subTitleSection(),
                    exampleSentenceSection(_currentState),
                  ]),
                  Padding(
                      padding: const EdgeInsets.all(20), child: nextButton())
                ])));
  }
}
