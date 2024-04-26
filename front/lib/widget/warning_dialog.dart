import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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
          semanticsLabel: texts.warningMessage[warningObject],
          style: const TextStyle(
              color: colors.textColor,
              fontSize: 13,
              fontWeight: FontWeight.w500),
        ),
        actions: [
          Container(
            width: MediaQuery.of(context).size.width * 0.8,
            child: fullyRoundedRectangleButton(colors.buttonColor, '확인', () {
              HapticFeedback.lightImpact();
              Navigator.of(context).pop();
            })
          )
        ]);
  }
}
