import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:flutter/services.dart';

OutlinedButton outlinedRoundedRectangleButton(String buttonText, Function pressedFunc) {
  return OutlinedButton(
    style: OutlinedButton.styleFrom(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(69),
      ),
      side: const BorderSide(
        color: colors.textColor,
        width: 0.4
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
    style: const TextStyle(
      color: colors.textColor,
      fontSize: 13,
      fontWeight: FontWeight.w700,
    ),
  )
);

}