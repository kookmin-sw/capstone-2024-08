import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:flutter/material.dart';

class WarningDialog extends StatelessWidget {
  final String warningObject;

  const WarningDialog({super.key, required this.warningObject});

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30),
        ),
        semanticLabel: texts.warningMessage[warningObject],
        title: Text(
          texts.warningMessage[warningObject]!,
          style: TextStyle(
              color: colors.textColor,
              fontSize: fonts.plainText(context),
              fontWeight: FontWeight.w500),
        ),
        actions: [
            warningObject == 'logout' || warningObject == 'deleteUser'
            ? Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                outlinedRoundedRectangleButton('확인', () {
                  Navigator.of(context).pop(true);
                }),
                fullyRoundedRectangleButton(colors.buttonColor, '취소', () {
                  Navigator.of(context).pop(false);
                })
              ])
            : fullyRoundedRectangleButton(colors.buttonColor, '확인', () {
                Navigator.of(context).pop();
            })
        ]);
  }
}
