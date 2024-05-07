import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/record/record_detail.dart';
import 'package:capstone/screen/script/script_detail.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

Text _buildTitle(String title){
  return Text(
    title,
    semanticsLabel: title,
    textAlign: TextAlign.center,
    overflow: TextOverflow.ellipsis,
    maxLines: 2,
    softWrap: false,
    style: const TextStyle(
      color: colors.textColor,
      fontSize: 14,
      fontWeight: FontWeight.w500,
    ),
  );
}

Text _buildCategory(String category){
  return Text(
    category,
    semanticsLabel: category,
    style: const TextStyle(
      color: colors.textColor,
      fontSize: 13,
      fontWeight: FontWeight.w800
    ),
  );
}

Text _buildContent(String content){
  return Text(
    '+ $content',
    semanticsLabel: content,
    overflow: TextOverflow.ellipsis,
    maxLines: 1,
    softWrap: false,
    style: const TextStyle(
      color: colors.textColor,
      fontSize: 12,
      fontWeight: FontWeight.w300,
    ),
  );
}

Text _buildPrecision(int? precision){
  return Text(
    '$precision',
    softWrap: false,
    style: const TextStyle(
      color: colors.buttonColor,
      fontSize: 14,
      fontWeight: FontWeight.w800,
    ),
  );
}

Widget scriptListTile(BuildContext context, ScriptModel script, String route, String scriptType, {RecordModel? record}) {
  return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        route == 'record' ?
          Get.to(() => RecordDetail(script: script, record: record))
          : Get.to(() => ScriptDetail(script: script, scriptType: scriptType));
      },
      child: Stack(
        children: [
          Container(
            margin: const EdgeInsets.fromLTRB(0, 10, 20, 10),
            width: MediaQuery.of(context).size.width * 0.85,
            height: MediaQuery.of(context).size.width * 0.45,
            decoration: ShapeDecoration(
                color: colors.blockColor,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(13),
                )),
            child: route == 'record' ?
            Container(
              padding: const EdgeInsets.fromLTRB(15, 15, 15, 0),
              child: Column(
                children: [
                  Align(
                    alignment: Alignment.topRight,
                    child: _buildPrecision(record!.promptResult!.last['precision'])
                  ),
                  const SizedBox(height: 30),
                  _buildTitle(script.title)
              ])
            )
            : Container(
                padding: const EdgeInsets.fromLTRB(15, 65, 15, 0),
                child: _buildTitle(script.title)
              ),
          ),
          Positioned(
            bottom: 0,
            left: 0,
            child: Container(
              margin: const EdgeInsets.fromLTRB(0, 10, 20, 10),
              width: MediaQuery.of(context).size.width * 0.85,
              height: MediaQuery.of(context).size.width * 0.18,
              decoration: ShapeDecoration(
                  color: colors.exampleScriptColor,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(13),
                  )),
              child: Container(
                margin: const EdgeInsets.fromLTRB(15, 0, 15, 0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildCategory(script.category),
                    _buildContent(script.content.join(' '))
                  ])
              )
            ))
      ])
  );
}