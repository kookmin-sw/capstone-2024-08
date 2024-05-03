import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

Container scriptContentBlock(String sentence, var width) {
  return Container(
      width: width,
      margin: const EdgeInsets.only(bottom: 20),
      decoration: BoxDecoration(
        color: colors.blockColor,
        borderRadius: BorderRadius.circular(10),
        boxShadow: const [
          BoxShadow(
            color: colors.buttonSideColor,
            blurRadius: 5,
            spreadRadius: 3,
          )
        ]),
      padding: const EdgeInsets.all(15),
      child: Text(
        sentence,
        semanticsLabel: sentence,
        textAlign: TextAlign.start,
        style: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w500,
          color: colors.textColor
        ),
      ),
  );
}