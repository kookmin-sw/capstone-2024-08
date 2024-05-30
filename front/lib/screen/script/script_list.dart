import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/model/load_data.dart';
import 'package:capstone/widget/category_buttons.dart';
import 'package:capstone/widget/script/create_user_script_button.dart';
import 'package:capstone/widget/script/read_script.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';

class ScriptList extends StatefulWidget {
  const ScriptList({Key? key, required this.index}) : super(key: key);

  final int index;

  @override
  State<ScriptList> createState() => _ScriptListState();
}

class _ScriptListState extends State<ScriptList> {
  final LoadData loadData = LoadData();
  String selectedCategoryValue = texts.category[0];

  void _handleCategorySelected(String value) {
    setState(() {
      selectedCategoryValue = value;
    });
  }

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);

    return Scaffold(
        body: Container(
            color: colors.bgrDarkColor,
            child: Column(children: [
              Container(
                padding: const EdgeInsets.fromLTRB(20, 10, 0, 15),
                child: CategoryButtons(
                    onCategorySelected: _handleCategorySelected),
              ),
              widget.index == 0
                  ? Expanded(
                      child: Padding(
                      padding: EdgeInsets.only(left: deviceWidth * 0.075),
                      child: readScripts(
                            loadData.readExampleScripts(selectedCategoryValue),
                            'example'))
                    )
                  : Expanded(
                      child: Padding(
                      padding: EdgeInsets.only(left: deviceWidth * 0.075),
                      child: Stack(children: [
                        readScripts(
                            loadData.readUserScripts(selectedCategoryValue),
                            'user'),
                        Positioned(
                            bottom: 3,
                            width: deviceWidth * 0.85,
                            child: createUserScriptButton(context))
                      ]))
                  )
            ])));
  }
}
