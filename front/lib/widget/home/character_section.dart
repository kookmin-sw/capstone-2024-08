import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/constants/image.dart' as images;
import 'package:flutter/material.dart';

Widget characterSection(BuildContext context, character) {
  return Container(
      decoration: BoxDecoration(
          color: colors.themeWhiteColor,
          borderRadius: BorderRadius.circular(20)),
      margin: const EdgeInsets.all(10),
      padding: const EdgeInsets.all(15),
      width: MediaQuery.of(context).size.width / 1.7,
      child: Image.asset(images.characterImagePaths[character]!));
}
