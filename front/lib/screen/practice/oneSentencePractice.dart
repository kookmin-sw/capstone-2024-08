import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/model/load_data.dart';
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/screen/practice/guide_voice_player.dart';
import 'package:capstone/widget/audio_recoder/recording_section.dart';
import 'package:capstone/widget/practice/pratice_app_bar.dart';
import 'package:capstone/widget/practice/scrap_button.dart';
import 'package:capstone/widget/progress_bar_section.dart';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';

class OneSentencePratice extends StatefulWidget {
  OneSentencePratice(
      {Key? key, required this.script, required this.scriptType, this.record})
      : super(key: key);

  final ScriptModel script;
  final String scriptType;
  RecordModel? record;

  @override
  State<OneSentencePratice> createState() => _OneSentencePraticeState();
}

class _OneSentencePraticeState extends State<OneSentencePratice> {
  LoadData loadData = LoadData();
  int _currentSentenceIndex = 0;
  int? sentenceLength;
  String uid = Get.find<UserController>().userModel.id!;

  double _currentProgressValue = 5;
  bool showPlayer = false;
  String? audioPath;
  String? currnetPracticeAudioPath;
  int? currentPrecision;
  List<int>? scrapSentences;

  @override
  void initState() {
    showPlayer = false;
    sentenceLength = widget.script.content.length;
    _currentSentenceIndex = 0;
    _currentProgressValue = (_currentSentenceIndex == 0)
        ? 5
        : (100 * _currentSentenceIndex / sentenceLength!);
    scrapSentences = widget.record == null ? [] : widget.record!.scrapSentence;
    super.initState();
  }

  void updateScrap(List<int>? updatedScrapSentences) {
    setState(() {
      scrapSentences = updatedScrapSentences;
    });
  }

  bool isClicked(int sentenceIndex) {
    if (scrapSentences == null) {
      return false;
    } else {
      if (scrapSentences!.contains(sentenceIndex)) {
        return true;
      }
      return false;
    }
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
      _currentSentenceIndex += 1;
      currnetPracticeAudioPath = audioPath!;
      _currentProgressValue = 100 * _currentSentenceIndex / sentenceLength!;
    });

    // if (_currentState == 'end') {
    //   debugPrint('Before: ${widget.userData.voiceUrls}');
    //   await uploadWavFilesToStorage();
    //   debugPrint('After: ${widget.userData.voiceUrls}');
    //   debugPrint(
    //       "이거 해야함 -> AuthController.instance.handleUserInfoCompletion()");
    //   await saveData.saveUserInfo(
    //       nickname: widget.userData.nickname!,
    //       character: widget.userData.character!,
    //       lastAccessDate: Timestamp.now(),
    //       voiceUrls: widget.userData.voiceUrls,
    //       attendanceStreak: 1);
    //   AuthController.instance.handleUserInfoCompletion();
    // }
  }

  Widget sentenceSection(int sentenceIndex) {
    return AnimatedSwitcher(
        duration: Duration(milliseconds: 300),
        child: Container(
          width: MediaQuery.of(context).size.width / 1.2,
          key: ValueKey<int>(sentenceIndex),
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
                alignment: Alignment.topLeft,
                padding: EdgeInsets.fromLTRB(5, 20, 5, 10),
                child: scrapsButton(widget.scriptType, widget.script.id!, uid,
                    sentenceIndex, isClicked(sentenceIndex), updateScrap)),
            Container(
              padding: EdgeInsets.fromLTRB(5, 20, 5, 20),
              child: Text(
                widget.script.content[sentenceIndex],
                textAlign: TextAlign.start,
                style: TextStyle(fontSize: 14.0),
              ),
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

  Future<Widget> precisionSection() async {
    if (showPlayer) {
      setState(() {
        currnetPracticeAudioPath = audioPath!;
      });
      currentPrecision = await getVoicesSimilarity(
          widget.script.content[_currentSentenceIndex],
          currnetPracticeAudioPath!);
      return Text('정확도 : $currentPrecision');
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

  Widget guideVoicePlayer() {
    return Container();
  }

  Widget waitingGetPrecisionSection(snapshot) {
    if (snapshot.connectionState == ConnectionState.waiting) {
      // 데이터 로딩 중일 때 표시할 위젯
      return const Column(children: [
        CircularProgressIndicator(
          color: colors.recordButtonColor,
        ),
        SizedBox(height: 20),
        Text(
          '유사도 계산하는 중',
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: practiceAppBar(),
        body: Stack(children: [
          Container(
              padding: const EdgeInsets.fromLTRB(20, 5, 20, 20),
              child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Column(children: [
                      _buildCategory(widget.script.category),
                      const SizedBox(height: 15),
                      _buildTitle(widget.script.title),
                      const SizedBox(height: 20),
                      progressBarSection(_currentProgressValue),
                    ]),
                    Container(
                        child: Column(children: [
                      Column(children: [
                        _currentSentenceIndex != sentenceLength
                            ? sentenceSection(_currentSentenceIndex)
                            : Container(),
                      ]),
                    ])),
                    FutureBuilder<Widget>(
                      future:
                          precisionSection(), // precisionSection 함수를 호출하여 Future<Widget>을 얻음
                      builder: (context, snapshot) {
                        return waitingGetPrecisionSection(snapshot);
                      },
                    ),
                    Padding(
                        padding: const EdgeInsets.all(20), child: nextButton())
                  ])),
        ]));
  }
}
