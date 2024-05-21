import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/fonts.dart' as fonts;

AppBar basicAppBar(
    {String title = '',
    Color backgroundColor = colors.bgrDarkColor,
    bool backButton = true}) {
  Color itemColor = backgroundColor == colors.bgrDarkColor
      ? colors.blockColor
      : colors.textColor;
  return AppBar(
      backgroundColor: backgroundColor,
      elevation: 0,
      centerTitle: true,
      title: Text(
        title,
        semanticsLabel: title,
        style: TextStyle(
            color: itemColor, fontSize: fonts.title, fontWeight: FontWeight.w700),
      ),
      leading: backButton
          ? IconButton(
              icon: Icon(Icons.keyboard_backspace_rounded, color: itemColor),
              onPressed: () => Get.back())
          : null);
}
