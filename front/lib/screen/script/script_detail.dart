import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/model/script.dart';
import 'package:capstone/widget/script/script_content_block.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:capstone/screen/script/select_practice.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class ScriptDetail extends StatefulWidget {
  const ScriptDetail({Key? key, required this.script}) : super(key: key);

  final ScriptModel script;
  final Color backgroundColor = colors.bgrBrightColor;

  @override
  State<ScriptDetail> createState() => _ScriptDetailState();
}

class _ScriptDetailState extends State<ScriptDetail> {
  Text _buildCategory(String category) {
    return Text(
      category,
      semanticsLabel: category,
      textAlign: TextAlign.start,
      style: const TextStyle(
          fontSize: 12, fontWeight: FontWeight.w500, color: colors.textColor),
    );
  }

  Text _buildTitle(String title) {
    return Text(
      title,
      semanticsLabel: title,
      textAlign: TextAlign.start,
      style: const TextStyle(
          fontSize: 15, fontWeight: FontWeight.w800, color: colors.textColor),
    );
  }

  @override
  Widget build(BuildContext context) {
    var width = MediaQuery.of(context).size.width;

    return Scaffold(
        appBar: basicAppBar(backgroundColor: widget.backgroundColor, title: ''),
        body: Stack(children: [
          Container(
              color: widget.backgroundColor,
              padding: const EdgeInsets.fromLTRB(20, 5, 20, 20),
              child: ListView(children: [
                _buildCategory(widget.script.category),
                const SizedBox(height: 15),
                _buildTitle(widget.script.title),
                const SizedBox(height: 20),
                scriptContentBlock(widget.script.content, width),
                const SizedBox(height: 30),
              ])),
          Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                  padding: const EdgeInsets.all(5),
                  decoration:
                      const BoxDecoration(color: colors.blockColor, boxShadow: [
                    BoxShadow(
                      color: colors.buttonSideColor,
                      blurRadius: 5,
                      spreadRadius: 5,
                    )
                  ]),
                  child: Container(
                      padding: const EdgeInsets.fromLTRB(20, 5, 20, 5),
                      child: false ?
                      fullyRoundedRectangleButton(colors.textColor, '연습하기', () {})
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Container(
                                width: width * 0.4,
                                child: outlinedRoundedRectangleButton('기록보기', () {})),
                            Container(
                              width: width * 0.4,
                              child: fullyRoundedRectangleButton(colors.buttonColor, '연습하기', () {
                                Get.to(() => SelectPractice(
                                  script: widget.script,
                                  tapCloseButton: () { Get.back(); },
                                ));
                              })
                            ),
                        ])
                    )
                ))
          ])
      );
  }
}
