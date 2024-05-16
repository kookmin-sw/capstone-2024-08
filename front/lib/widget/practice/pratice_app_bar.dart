import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:capstone/constants/color.dart' as colors;

AppBar practiceAppBar() {
  return AppBar(
      backgroundColor: colors.bgrBrightColor,
      elevation: 0,
      leading: IconButton(
          icon: const Icon(Icons.keyboard_backspace_rounded,
              color: colors.textColor),
          onPressed: () => Get.back()));
}
