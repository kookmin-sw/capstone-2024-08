import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/widget/record/read_record_script.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/model/load_data.dart';
import 'package:capstone/widget/category_buttons.dart';


class RecordList extends StatefulWidget {
  const RecordList({
    Key? key,
    required this.index
  }) : super(key: key);

  final int index;

  @override
  State<RecordList> createState() => _RecordListState();
}

class _RecordListState extends State<RecordList> {
  final LoadData loadData = LoadData();
  String selectedCategoryValue = texts.category[0];

  void _handleCategorySelected(String value) {
    setState(() {
      selectedCategoryValue = value;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Container(
          padding: const EdgeInsets.fromLTRB(20, 10, 0, 0),
          color: colors.bgrDarkColor,
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.fromLTRB(0, 0, 0, 15),
                child: CategoryButtons(onCategorySelected: _handleCategorySelected),
              ),
              widget.index == 0 ?
                Expanded(
                  child: readRecordScripts(loadData.readExampleScripts(selectedCategoryValue), 'example')
                )
                : Expanded(
                    child: readRecordScripts(loadData.readUserScripts(selectedCategoryValue), 'user'),                     
                  )                    
          ])
        )
    );
  }
}
