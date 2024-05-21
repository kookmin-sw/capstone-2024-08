import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/record/prompt_precision_graph.dart';
import 'package:capstone/screen/record/scrap_sentence_slider.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/screen/script/select_practice.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class RecordDetail extends StatefulWidget {
  RecordDetail({
    Key? key,
    required this.script,
    required this.record,
    required this.scriptType
  }) : super(key: key);

  ScriptModel script;
  RecordModel? record;
  String scriptType;

  @override
  State<RecordDetail> createState() => _RecordDetailState();
}

class _RecordDetailState extends State<RecordDetail> {
  List<String> scrapSentenceList = [];

  @override
  void initState() {
    _checkScrapList();
    super.initState();
  }

  void _checkScrapList() {
    for (int idx = 0; idx < widget.script.content.length; idx++) {
      if (widget.record!.scrapSentence != null) {
        if (widget.record!.scrapSentence!.contains(idx)) {
          scrapSentenceList.add(widget.script.content[idx]);
        }
      }
    }
  }

  Text _buildText(String text, double fontSize) {
    return Text(
      text,
      semanticsLabel: text,
      textAlign: TextAlign.start,
      style: TextStyle(
          fontSize: fontSize,
          fontWeight: FontWeight.w500,
          color: colors.textColor),
    );
  }

  Text _buildCategory(String category) {
    return _buildText(category, fonts.category);
  }

  Text _buildTitle(String title) {
    return Text(
      title,
      semanticsLabel: title,
      textAlign: TextAlign.start,
      style: const TextStyle(
          fontSize: fonts.title,
          fontWeight: FontWeight.w600,
          color: colors.textColor),
    );
  }

  Text _buildRecordItemTitle(String title) {
    return _buildText(title, fonts.plainText);
  }

  Container _notExistsRecord(String item) {
    return Container(
        padding: EdgeInsets.only(
            top: getDeviceHeight(context) * 0.1,
            bottom: getDeviceHeight(context) * 0.1),
        child: Align(alignment: Alignment.center, child: _buildText(item, fonts.plainText)));
  }

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);
    var deviceHeight = getDeviceHeight(context);

    return Scaffold(
        appBar: basicAppBar(title: '기록'),
        body: Stack(children: [
          ListView(children: [
            Container(
                width: deviceWidth,
                padding: const EdgeInsets.all(20),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildCategory(widget.script.category),
                      SizedBox(height: deviceHeight * 0.01),
                      _buildTitle(widget.script.title),
                      SizedBox(height: deviceHeight * 0.04),
                      _buildRecordItemTitle('스크랩한 문장 목록'),
                      scrapSentenceList.isNotEmpty
                          ? ScrapSentenceSlider(
                              scrapSentenceList: scrapSentenceList)
                          : _notExistsRecord('스크랩한 문장이 존재하지 않습니다.'),
                      SizedBox(height: deviceHeight * 0.04),
                      _buildRecordItemTitle('프롬프트 정확도 추이 그래프'),
                      widget.record!.promptResult!.isNotEmpty
                          ? PromptPrecisionGraph(
                              promptResult: widget.record!.promptResult!)
                          : _notExistsRecord('프롬프트 연습 기록이 존재하지 않습니다.'),
                      SizedBox(height: deviceHeight * 0.05),
                    ]))
          ]),
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
                      child: fullyRoundedRectangleButton(
                          colors.textColor, '다시 연습하기', () {
                          Get.to(() => SelectPractice(
                            script: widget.script,
                            tapCloseButton: () { Get.back(); },
                            scriptType: widget.scriptType,
                            record: widget.record
                          ));
                      }))))
        ]));
  }
}
