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

  nextButtonPressed(double value) {
    setState(() {
      _currentProgressValue = value;
    });
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
            RecordingSection()
          ]),
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
            child: Column(children: [
              progressBarSection(_currentProgressValue),
              subTitleSection(),
              exampleSentenceSection('long'),
            ])));
  }
}
