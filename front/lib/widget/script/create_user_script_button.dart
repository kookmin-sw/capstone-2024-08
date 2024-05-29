import 'package:capstone/screen/script/create_user_script/create_user_script.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:flutter/services.dart';
import 'package:get/get.dart';

ElevatedButton createUserScriptButton(BuildContext context) {
  return ElevatedButton(
      style: ElevatedButton.styleFrom(
        backgroundColor: colors.buttonColor,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
      ),
  onPressed: () {
    HapticFeedback.lightImpact();
    Get.to(() => const CreateUserScript());
  },
  child: Text(
    '나만의 대본 만들기',
    semanticsLabel: '나만의 대본 만들기',
    textAlign: TextAlign.center,
    style: TextStyle(
      color: colors.blockColor,
      fontSize: fonts.plainText(context),
      fontWeight: FontWeight.w700,
    ),
  )
);
}
