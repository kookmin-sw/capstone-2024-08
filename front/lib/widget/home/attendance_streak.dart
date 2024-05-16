import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/constants/color.dart' as colors;

Widget attendanceStreakSection(name, day) {
  return Container(
    decoration: BoxDecoration(
        color: colors.themeWhiteColor, borderRadius: BorderRadius.circular(20)),
    margin: const EdgeInsets.all(15),
    child: ListTile(
      title: Text(
        '$day일',
        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 20.0),
      ),
      subtitle: Text(
        '$name ${texts.attendaceStreakMessage}',
        style: const TextStyle(fontWeight: FontWeight.normal),
      ),
      leading: const Icon(
        CupertinoIcons.heart_circle_fill,
        size: 45,
        color: colors.bgrDarkColor,
      ),
    ),
  );
}
