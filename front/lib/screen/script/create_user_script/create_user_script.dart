import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/screen/script/create_user_script/adjust_user_script.dart';
import 'package:capstone/screen/script/create_user_script/category_section.dart';
import 'package:capstone/screen/script/create_user_script/content_controller.dart';
import 'package:capstone/screen/script/create_user_script/content_section.dart';
import 'package:capstone/screen/script/create_user_script/title_section.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/bottom_buttons.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:capstone/widget/warning_dialog.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';


class CreateUserScript extends StatefulWidget {
  const CreateUserScript({Key? key}) : super(key: key);

  final Color backgroundColor = colors.bgrBrightColor;

  @override
  State<CreateUserScript> createState() => _CreateUserScriptState();
}

class _CreateUserScriptState extends State<CreateUserScript> {
  final TextEditingController _title = TextEditingController();
  final TextEditingController _content = TextEditingController();
  String? _selectedCategory;
  final myFocus = FocusNode();
  final _formkey = GlobalKey<FormState>();

  void _handleCategorySelected(String value) {
    setState(() {
      _selectedCategory = value;
    });
  }

 
  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;

    return Scaffold(
      appBar: basicAppBar(title: '나만의 대본 만들기'),
      body: Stack(
        children: [
          Container(
            color: widget.backgroundColor,
            child: Form(
                key: _formkey,
                child: Column(
                  children: [
                    TitleSection(titleController: _title, focus: myFocus),
                    CategorySection(onCategorySelected: _handleCategorySelected),
                    ContentSection(contentController: _content, focus: myFocus)
                ])
            )
          ),
          bottomButtons(
            width, 
            outlinedRoundedRectangleButton('AI로 생성하기', () {}), 
            fullyRoundedRectangleButton(colors.buttonColor, '완료', () { 
              if (_selectedCategory == null) {
                showDialog(
                    context: context,
                    builder: (context) {
                      return const WarningDialog(
                          warningObject: 'category');
                    });
              }

              if (_formkey.currentState!.validate()) {
                List<String>? sentenceList = splitContent(_content.text);
                Get.put(UserScriptContentController(sentenceList));

                Get.to(() => AdjustUserScript(
                  title: _title.text, 
                  category: _selectedCategory!,
                ));
              }
            })
          ),
      ])        
    );
  }
}

List<String> splitContent(String content) {
  List<String> sentenceList = [];
  for(String sentence in content.split('.')){
    sentence.trim();
    sentenceList.add('$sentence.');
  }
  if(sentenceList.last == '.') { sentenceList.removeLast(); }
  return sentenceList;
}