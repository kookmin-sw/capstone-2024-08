import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/model/load_data.dart';
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/record/record_detail.dart';
import 'package:capstone/widget/audio_recoder/recording_section.dart';
import 'package:capstone/widget/practice/pratice_app_bar.dart';
import 'package:capstone/widget/script/script_content_block.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:capstone/screen/script/select_practice.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class OneSentencePratice extends StatefulWidget {
  const OneSentencePratice(
      {Key? key,
      required this.script,
      required this.scriptType,
      required this.recordExists})
      : super(key: key);

  final ScriptModel script;
  final String scriptType;
  final bool recordExists;

  @override
  State<OneSentencePratice> createState() => _OneSentencePraticeState();
}

class _OneSentencePraticeState extends State<OneSentencePratice> {
  LoadData loadData = LoadData();
  RecordModel? record;
  int _currentSentenceIndex = 0;
  int? sentenceLength;
  String buttonText = '연습하기';

  double _currentProgressValue = 5;
  bool showPlayer = false;
  String? audioPath;

  @override
  void initState() {
    super.initState();
    showPlayer = false;
    buttonText = widget.recordExists ? '다시 연습하기' : '연습하기';
    sentenceLength = widget.script.content.length;
    _currentProgressValue = 100 * _currentSentenceIndex / sentenceLength!;
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: PracticeAppBar(
          script: widget.script,
          scriptType: widget.scriptType,
          backButton: true,
          scrapSentences: [1, 2],
          sentenceIndex: _currentSentenceIndex,
        ),
        body: Stack(children: [
          Container(
              padding: const EdgeInsets.fromLTRB(20, 5, 20, 20),
              child: ListView(children: [
                _buildCategory(widget.script.category),
                const SizedBox(height: 15),
                _buildTitle(widget.script.title),
                const SizedBox(height: 20),
                Container(
                    color: colors.bgrBrightColor,
                    child: Column(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Column(children: [
                            progressBarSection(_currentProgressValue),
                            subTitleSection(),
                            _currentState != 'end'
                                ? sentenceSection(_currentState)
                                : Container(),
                          ]),
                          Padding(
                              padding: const EdgeInsets.all(20),
                              child: nextButton())
                        ]))
              ])),
          Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                  padding: const EdgeInsets.all(5),
                  decoration:
                      const BoxDecoration(color: colors.blockColor, boxShadow: [
                    BoxShadow(
                      color: colors.buttonSideColor,
                      blurRadius: 5,
                      spreadRadius: 5,
                    )
                  ]),
                  child: Container(
                      padding: const EdgeInsets.fromLTRB(20, 5, 20, 5),
                      child: recordExists
                          ? Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                  Container(
                                      width: getDeviceWidth(context) * 0.4,
                                      child: outlinedRoundedRectangleButton(
                                          '기록보기', () async {
                                        Get.to(() => RecordDetail(
                                            script: widget.script,
                                            record: record));
                                      })),
                                  Container(
                                      width: getDeviceWidth(context) * 0.4,
                                      child: fullyRoundedRectangleButton(
                                          colors.buttonColor, '연습하기', () {
                                        Get.to(() => SelectPractice(
                                              script: widget.script,
                                              tapCloseButton: () {
                                                Get.back();
                                              },
                                            ));
                                      })),
                                ])
                          : fullyRoundedRectangleButton(
                              colors.textColor, '연습하기', () {
                              Get.to(() => SelectPractice(
                                    script: widget.script,
                                    tapCloseButton: () {
                                      Get.back();
                                    },
                                  ));
                            }))))
        ]));
  }
}
