import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:flutter/material.dart';

class PromptResult extends StatefulWidget {
  PromptResult(
      {super.key,
      required this.script,
      required this.scriptType,
      this.guideVoicePath,
      this.record});

  final ScriptModel script;
  final String scriptType;
  final String? guideVoicePath;
  RecordModel? record;

  @override
  State<PromptResult> createState() => _PromptResultState();
}

class _PromptResultState extends State<PromptResult> {
  @override
  Widget build(BuildContext context) {
    return const Placeholder();
  }
}
