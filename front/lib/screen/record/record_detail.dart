import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/record/scrap_sentence_slider.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/screen/script/select_practice.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';


class RecordDetail extends StatefulWidget {
  RecordDetail({
    Key? key,
    required this.script,
    required this.record,
  }) : super(key: key);

  ScriptModel script;
  RecordModel? record;
  final Color backgroundColor = colors.bgrBrightColor;

  @override
  State<RecordDetail> createState() => _RecordDetailState();
}

class _RecordDetailState extends State<RecordDetail> {
  List<String> scrapSentenceList = [];

  @override
  void initState() {
    super.initState();
    _checkScrapList();
  }

  void _checkScrapList() {
    for(int idx=0; idx<widget.script.content.length; idx++){
      if(widget.record!.scrapSentence.contains(idx)) {
        scrapSentenceList.add(widget.script.content[idx]);
      }
    }
  }

  Text _buildCategory(String category){
    return Text(
      category,
      semanticsLabel: category,
      textAlign: TextAlign.start,
      style: const TextStyle(
        fontSize: 12,
        fontWeight: FontWeight.w500,
        color: colors.textColor
      ),
    );
  }

  Text _buildTitle(String title){
    return Text(
      title,
      semanticsLabel: title,
      textAlign: TextAlign.start,
      style: const TextStyle(
        fontSize: 15,
        fontWeight: FontWeight.w500,
        color: colors.textColor
      ),
    );
  }

  Text _notExistsScrapSentence() {
    return const Text(
      '스크랩한 문장이\n존재하지 않습니다.',
      semanticsLabel: '스크랩한 문장이 존재하지 않습니다.',
      textAlign: TextAlign.center,
      style: TextStyle(
        fontSize: 15,
        fontWeight: FontWeight.w500,
        color: colors.textColor
      ),
    );
  }

  Text _buildRecordItemTitle(String title) {
    return Text(
      title,
      semanticsLabel: title,
      textAlign: TextAlign.start,
      style: const TextStyle(
        fontSize: 12,
        fontWeight: FontWeight.w500,
        color: colors.textColor
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: basicAppBar(title: '기록'),
        body: Stack(
          children: [
            Container(
              color: widget.backgroundColor,
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildCategory(widget.script.category),
                  const SizedBox(height: 12),
                  _buildTitle(widget.script.title),
                  const SizedBox(height: 20),
                  _buildRecordItemTitle('스크랩한 문장 목록'),
                  scrapSentenceList.isNotEmpty ?
                    ScrapSentenceSlider(scrapSentenceList: scrapSentenceList)
                    : _notExistsScrapSentence(),
                  const SizedBox(height: 20),
                  _buildRecordItemTitle('프롬프트 정확도 추이 그래프'),
                ])),    
                Positioned(
                  left: 0,
                  right: 0,
                  bottom: 0,
                  child: Container(
                    padding: const EdgeInsets.all(5),
                    decoration:
                      const BoxDecoration(
                        color: colors.blockColor, 
                        boxShadow: [
                          BoxShadow(
                            color: colors.buttonSideColor,
                            blurRadius: 5,
                            spreadRadius: 5,
                          )
                      ]),
                    child: Container(
                      padding: const EdgeInsets.fromLTRB(20, 5, 20, 5),
                      child: fullyRoundedRectangleButton(colors.textColor, '다시 연습하기', () {
                        Get.to(() => SelectPractice(
                          script: widget.script,
                          tapCloseButton: () { Get.back(); },
                        ));
                      })
                    )
                ))
          ])
      );
  }
}
