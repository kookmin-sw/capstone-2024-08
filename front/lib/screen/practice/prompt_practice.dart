import 'dart:async';
import 'dart:io';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:get/get.dart';

class PromptPractice extends StatefulWidget {
  PromptPractice({super.key, required this.script});

  final ScriptModel script;
  @override
  State<PromptPractice> createState() => _PromptPracticeState();
}

class _PromptPracticeState extends State<PromptPractice> {
  final ScrollController _scrollController = ScrollController();
  final Map<String, File?> _wavFiles = Get.find<UserController>().wavFiles;

  @override
  void initState() {
    super.initState();
    Timer.periodic(Duration(milliseconds: 1000), (Timer timer) {
      // 스크롤이 더 내려갈 수 있는지 확인
      if (_scrollController.offset <
          _scrollController.position.maxScrollExtent) {
        // 한 픽셀씩 아래로 스크롤
        _scrollController.animateTo(
          _scrollController.offset + 50,
          duration: Duration(milliseconds: 300),
          curve: Curves.ease,
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
