import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:flutter/services.dart';

ElevatedButton fullyRoundedRectangleButton(Color backgroundColor, String buttonText, Function pressedFunc) {
  return ElevatedButton(
      style: ElevatedButton.styleFrom(
        backgroundColor: backgroundColor,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(69),
        ),
      ),
  onPressed: () {
    HapticFeedback.lightImpact();
    pressedFunc();
  },
  child: Text(
    buttonText,
    semanticsLabel: buttonText,
    textAlign: TextAlign.center,
    style: TextStyle(
      color: backgroundColor != colors.blockColor 
        ? colors.blockColor
        : colors.textColor,
      fontSize: fonts.button,
      fontWeight: FontWeight.w700,
    ),
  )
);

}
