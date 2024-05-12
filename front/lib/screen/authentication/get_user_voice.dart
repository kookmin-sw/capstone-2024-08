import 'package:capstone/model/save_data.dart';
import 'package:capstone/screen/authentication/controller/auth_controller.dart';
import 'package:capstone/widget/audio_recoder/recording_section.dart';
import 'package:capstone/model/user.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animation_progress_bar/flutter_animation_progress_bar.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:flutter/widgets.dart';

class GetUserVoice extends StatefulWidget {
  final UserModel userData;
  const GetUserVoice({super.key, required this.userData});

  @override
  State<GetUserVoice> createState() => _GetUserVoiceState();
}

class _GetUserVoiceState extends State<GetUserVoice> {
  final SaveData saveData = SaveData();

  double _currentProgressValue = 5;
  String _currentState = 'short';
  bool showPlayer = false;
  String? audioPath;

  @override
  void initState() {
    showPlayer = false;
    super.initState();
  }

  String getNextState(String currentState) {
    if (currentState == 'short') {
      return 'middle';
    } else if (currentState == 'middle') {
      return 'long';
    } else {
      return 'end';
    }
  }

  Future<void> uploadWavFilesToStorage() async {
    widget.userData.voiceUrls = await saveData.uploadWavFiles(
        widget.userData.id!, widget.userData.voiceUrls!);
  }

  nextButtonPressed(String currentState) async {
    setState(() {
      widget.userData.voiceUrls?[currentState] = audioPath!;
      showPlayer = false;
      _currentState = getNextState(currentState);
      _currentProgressValue = texts.getUserProgressValues[_currentState]!;
    });

    if (_currentState == 'end') {
      debugPrint('Before: ${widget.userData.voiceUrls}');
      await uploadWavFilesToStorage();
      debugPrint('After: ${widget.userData.voiceUrls}');
      debugPrint(
          "이거 해야함 -> AuthController.instance.handleUserInfoCompletion()");
      await saveData.saveUserInfo(
          nickname: widget.userData.nickname!,
          character: widget.userData.character!,
          lastAccessDate: Timestamp.now(),
          voiceUrls: widget.userData.voiceUrls,
          attendanceStreak: 1);
      AuthController.instance.handleUserInfoCompletion();
    }
  }

  Widget progressBarSection(double currentValue) {
    return Container(
        padding: const EdgeInsets.fromLTRB(15, 15, 20, 20),
        color: colors.bgrBrightColor,
        child: FAProgressBar(
          currentValue: currentValue,
          animatedDuration: Duration(milliseconds: 500),
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
        duration: Duration(milliseconds: 300),
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
          onPressed: () {
            debugPrint('녹음 완료 상태 : $showPlayer');
            if (showPlayer) {
              nextButtonPressed(_currentState);
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
                  colors.bgrDarkColor)), // 값을 변경하도록 수정
          child: Text(
            texts.nextButtonText,
            style: const TextStyle(
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
                    _currentState != 'end'
                        ? exampleSentenceSection(_currentState)
                        : Container(),
                  ]),
                  Padding(
                      padding: const EdgeInsets.all(20), child: nextButton())
                ])));
  }
}
