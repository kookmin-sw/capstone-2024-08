import 'dart:async';
import 'dart:io';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:get/get.dart';

class PromptPractice extends StatefulWidget {
  PromptPractice({super.key, required this.script, this.guideVoicePath});

  final ScriptModel script;
  final String? guideVoicePath;

  @override
  State<PromptPractice> createState() => _PromptPracticeState();
}

class _PromptPracticeState extends State<PromptPractice> {
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    Timer.periodic(Duration(milliseconds: 500), (Timer timer) {
      // 스크롤이 더 내려갈 수 있는지 확인
      if (_scrollController.hasClients) {
        // 한 픽셀씩 아래로 스크롤
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(seconds: 8),
          curve: Curves.easeIn,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: colors.textColor,
      body: ListView.builder(
        controller: _scrollController,
        itemCount: widget.script.content.length, // 텍스트 아이템의 개수
        itemBuilder: (BuildContext context, int index) {
          // 텍스트 아이템 생성
          return ListTile(
            title: Text(
              widget.script.content[index],
              style: TextStyle(color: colors.themeWhiteColor, fontSize: 40),
            ),
          );
        },
      ),
    );
  }
}
