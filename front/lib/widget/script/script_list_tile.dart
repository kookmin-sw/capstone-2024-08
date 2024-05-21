import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/record/record_detail.dart';
import 'package:capstone/screen/script/script_detail.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';
import 'package:capstone/constants/fonts.dart' as fonts;

Text _buildTitle(BuildContext context, String title) {
  return Text(
    title,
    semanticsLabel: title,
    textAlign: TextAlign.center,
    overflow: TextOverflow.ellipsis,
    maxLines: 1,
    softWrap: false,
    style: TextStyle(
      color: colors.textColor,
      fontSize: fonts.title(context),
      fontWeight: FontWeight.w300,
    ),
  );
}

Text _buildCategory(BuildContext context, String category) {
  return Text(
    category,
    semanticsLabel: category,
    style: TextStyle(
        color: colors.textColor, 
        fontSize: fonts.category(context) * 0.95, 
        fontWeight: FontWeight.w600
      ),
  );
}

Text _buildContent(BuildContext context, String content) {
  return Text(
    '+ $content',
    semanticsLabel: content,
    overflow: TextOverflow.ellipsis,
    maxLines: 1,
    softWrap: false,
    style: TextStyle(
      color: colors.textColor,
      fontSize: fonts.plainText(context) * 0.9,
      fontWeight: FontWeight.w300,
    ),
  );
}

Text _buildPrecision(BuildContext context, int? precision) {
  return Text(
    '$precision',
    softWrap: false,
    style: TextStyle(
      color: colors.buttonColor,
      fontSize: fonts.plainText(context),
      fontWeight: FontWeight.w800,
    ),
  );
}

Widget scriptListTile(
    BuildContext context, ScriptModel script, String route, String scriptType,
    {RecordModel? record}) {
  var deviceWidth = getDeviceWidth(context);

  return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        route == 'record'
            ? Get.to(() => RecordDetail(
                script: script, record: record, scriptType: scriptType))
            : Get.to(
                () => ScriptDetail(script: script, scriptType: scriptType));
      },
      child: Stack(children: [
        Container(
          margin: const EdgeInsets.fromLTRB(0, 10, 20, 10),
          width: deviceWidth * 0.85,
          height: deviceWidth * 0.45,
          decoration: ShapeDecoration(
              color: colors.blockColor,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(13),
              )),
          child: route == 'record'
              ? Column(children: [
                  record!.promptResult!.isNotEmpty
                      ? Align(
                          alignment: Alignment.topRight,
                          child: Padding(
                              padding: const EdgeInsets.fromLTRB(0, 10, 15, 0),
                              child: _buildPrecision(context,
                                  record.promptResult!.last['precision'])))
                      : const Padding(padding: EdgeInsets.only(bottom: 30)),
                  Padding(
                      padding: EdgeInsets.fromLTRB(15, deviceWidth * 0.07, 15, 0),
                      child: _buildTitle(context, script.title))
                ])
              : Padding(
                  padding: EdgeInsets.fromLTRB(15, deviceWidth * 0.12, 15, 0),
                  child: _buildTitle(context, script.title)),
        ),
        Positioned(
            bottom: 0,
            left: 0,
            child: Container(
                margin: const EdgeInsets.fromLTRB(0, 10, 20, 10),
                width: deviceWidth * 0.85,
                height: deviceWidth * 0.18,
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
                          _buildCategory(context, script.category),
                          _buildContent(context, script.content.join(' '))
                        ]))))
      ]));
}
