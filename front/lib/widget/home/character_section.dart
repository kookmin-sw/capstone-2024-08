import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/image.dart' as images;
import 'package:flutter/material.dart';

Widget characterSection(String character) {
  return Container(
      decoration: BoxDecoration(
          color: colors.themeWhiteColor,
          borderRadius: BorderRadius.circular(20)),
      margin: const EdgeInsets.all(10),
      padding: const EdgeInsets.all(20),
      child: Image.asset(images.characterImagePaths[character]!));
}
