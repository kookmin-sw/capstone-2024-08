import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:capstone/model/load_data.dart';
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/record/record_detail.dart';
import 'package:capstone/widget/script/script_content_block.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:capstone/screen/script/select_practice.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

class ScriptDetail extends StatefulWidget {
  const ScriptDetail({Key? key, required this.script, required this.scriptType})
      : super(key: key);

  final ScriptModel script;
  final String scriptType;

  @override
  State<ScriptDetail> createState() => _ScriptDetailState();
}

class _ScriptDetailState extends State<ScriptDetail> {
  LoadData loadData = LoadData();
  RecordModel? record;
  bool recordExists = false;

  @override
  void initState() {
    super.initState();
    checkRecordExists();
  }

  Future<void> checkRecordExists() async {
    DocumentSnapshot<Map<String, dynamic>> recordDocument =
        await loadData.readRecordDocument(widget.scriptType, widget.script.id!);

    if (recordDocument.exists) {
      setState(() {
        record = RecordModel.fromDocument(doc: recordDocument);
        recordExists = true;
      });
    }
  }

  Text _buildCategory(String category) {
    return Text(
      category,
      semanticsLabel: category,
      textAlign: TextAlign.start,
      style: const TextStyle(
          fontSize: fonts.category, 
          fontWeight: FontWeight.w500, 
          color: colors.textColor
      ),
    );
  }

  Text _buildTitle(String title) {
    return Text(
      title,
      semanticsLabel: title,
      textAlign: TextAlign.start,
      style: const TextStyle(
          fontSize: fonts.title, 
          fontWeight: FontWeight.w700, 
          color: colors.textColor
        ),
    );
  }

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);
    var deviceHeight = getDeviceHeight(context);
    
    return Scaffold(
        appBar: basicAppBar(backgroundColor: colors.bgrBrightColor, title: ''),
        body: Stack(children: [
          Container(
              padding: const EdgeInsets.fromLTRB(20, 5, 20, 20),
              child: ListView(children: [
                _buildCategory(widget.script.category),
                SizedBox(height: deviceHeight * 0.01),
                _buildTitle(widget.script.title),
                SizedBox(height: deviceHeight * 0.03),
                Column(
                    children: widget.script.content
                        .map((sentence) => scriptContentBlock(
                            sentence, deviceWidth))
                        .toList()),
                SizedBox(height: deviceHeight * 0.06),
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
                      child: recordExists
                          ? Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                  Container(
                                      width: deviceWidth * 0.4,
                                      child: outlinedRoundedRectangleButton(
                                          '기록보기', () async {
                                        Get.to(() => RecordDetail(
                                            script: widget.script,
                                            record: record,
                                            scriptType: widget.scriptType,
                                        ));
                                      })),
                                  Container(
                                      width: deviceWidth * 0.4,
                                      child: fullyRoundedRectangleButton(
                                          colors.buttonColor, '연습하기', () {
                                        Get.to(() => SelectPractice(
                                              script: widget.script,
                                              tapCloseButton: () {
                                                Get.back();
                                              },
                                              scriptType: widget.scriptType,
                                              record: record,
                                            ));
                                      })),
                                ])
                          : fullyRoundedRectangleButton(
                              colors.textColor, '연습하기', () {
                              Get.to(() => SelectPractice(
                                    script: widget.script,
                                    tapCloseButton: () {
                                      Get.back();
                                    },
                                    scriptType: widget.scriptType,
                                  ));
                            }))))
        ]));
  }
}
