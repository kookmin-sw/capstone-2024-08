import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animation_progress_bar/flutter_animation_progress_bar.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;

import 'package:flutter/widgets.dart';
import 'package:get/get.dart';
import 'package:get/get_state_manager/src/simple/get_widget_cache.dart';

class GetUserVoice extends StatefulWidget {
  const GetUserVoice({super.key});

  @override
  State<GetUserVoice> createState() => _GetUserVoiceState();
}

class _GetUserVoiceState extends State<GetUserVoice> {
  double _currentValue = 5;

  nextButtonPressed(double value) {
    setState(() {
      _currentValue = value;
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

  Widget exampleSentenceSection() {
    return Container();
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
              progressBarSection(_currentValue),
              subTitleSection()
            ])));
  }
}
