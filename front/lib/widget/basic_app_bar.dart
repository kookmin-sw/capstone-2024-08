import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:capstone/constants/color.dart' as colors;

AppBar basicAppBar({String title = '', Color backgroundColor = colors.bgrDarkColor}) {
  Color itemColor = backgroundColor == colors.bgrDarkColor ? colors.blockColor : colors.textColor;
    return AppBar(
      backgroundColor: backgroundColor,
      elevation: 0,
      centerTitle: true,
      title: Text(
        title,
        semanticsLabel: title,
          style: TextStyle(
            color: itemColor,
            fontSize: 20,
            fontWeight: FontWeight.w700
          ),
        ),
      leading: IconButton(
        icon: Icon(Icons.keyboard_backspace_rounded, color: itemColor),
        onPressed: () => Get.back()),
    );
  }