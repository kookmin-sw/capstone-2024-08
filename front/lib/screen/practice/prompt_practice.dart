import 'dart:async';
import 'package:capstone/model/script.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

class PromptPractice extends StatefulWidget {
  PromptPractice({super.key, required this.script});

  final ScriptModel script;
  @override
  State<PromptPractice> createState() => _PromptPracticeState();
}

class _PromptPracticeState extends State<PromptPractice> {
  final ScrollController _controller = ScrollController();

  @override
  void initState() {
    super.initState();
    Timer.periodic(Duration(milliseconds: 1000), (Timer timer) {
      // 스크롤이 더 내려갈 수 있는지 확인
      if (_controller.offset < _controller.position.maxScrollExtent) {
        // 한 픽셀씩 아래로 스크롤
        _controller.animateTo(
          _controller.offset + 1,
          duration: Duration(milliseconds: 500),
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
        controller: _controller,
        itemCount: widget.script.content.length, // 텍스트 아이템의 개수
        itemBuilder: (BuildContext context, int index) {
          // 텍스트 아이템 생성
          return ListTile(
            title: Text(
              widget.script.content[index],
              style: TextStyle(color: colors.themeWhiteColor, fontSize: 50),
            ),
          );
        },
      ),
    );
  }
}
