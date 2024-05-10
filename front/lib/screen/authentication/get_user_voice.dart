import 'package:capstone/model/user.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animation_progress_bar/flutter_animation_progress_bar.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';

class GetUserVoice extends StatefulWidget {
  GetUserVoice({
    super.key,
    required this.userData
  });

  UserModel userData;

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

  Widget exampleSentenceSection(String exampleSentence) {
    return AnimatedSwitcher(
        duration: Duration(milliseconds: 500),
        child: Container(
          key: ValueKey<String>(exampleSentence),
          padding: EdgeInsets.all(20.0),
          decoration: BoxDecoration(
            border: Border.all(color: colors.bgrDarkColor, width: 2.0),
            borderRadius: BorderRadius.circular(10.0),
          ),
          child: Text(
            // texts.getUserVoiceExampleSentences[exampleSentence],
            '',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16.0),
          ),
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
              subTitleSection()
            ])));
  }
}
