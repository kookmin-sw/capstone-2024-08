import 'package:flutter/widgets.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:flutter_animation_progress_bar/flutter_animation_progress_bar.dart';

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
