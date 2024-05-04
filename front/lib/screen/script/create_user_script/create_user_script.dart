import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/screen/script/create_user_script/adjust_user_script.dart';
import 'package:capstone/screen/script/create_user_script/input_field/category_section.dart';
import 'package:capstone/screen/script/create_user_script/controller/content_controller.dart';
import 'package:capstone/screen/script/create_user_script/input_field/content_section.dart';
import 'package:capstone/screen/script/create_user_script/controller/gpt_controller.dart';
import 'package:capstone/screen/script/create_user_script/input_field/title_section.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/bottom_buttons.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:capstone/widget/warning_dialog.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';


class CreateUserScript extends StatefulWidget {
  const CreateUserScript({Key? key}) : super(key: key);

  @override
  State<CreateUserScript> createState() => _CreateUserScriptState();
}

class _CreateUserScriptState extends State<CreateUserScript> {
  final TextEditingController _title = TextEditingController();
  final TextEditingController _content = TextEditingController();
  String? _selectedCategory;
  final _titleKey = GlobalKey<FormState>();
  final _contentKey = GlobalKey<FormState>();

  void _handleCategorySelected(String value) {
    setState(() {
      _selectedCategory = value;
    });
  }

  void createScriptByGpt(String title, String category) async {
      setState(() {
        _content.text = 'AI로 대본을 생성하고 있습니다. 잠시만 기다려주세요.';
      });
      await GptController().createScript(title, category).then((script) {
        setState(() {
          _content.text = script;
        });
      });
  }

  void checkValidCategory(String? category) {
    if (category == null) {
      showDialog(
        context: context,
        builder: (context) {
          return const WarningDialog(
              warningObject: 'category');
        });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: basicAppBar(title: '나만의 대본 만들기'),
      body: Stack(
        children: [
          Column(
            children: [
              TitleSection(titleController: _title, formKey: _titleKey),
              CategorySection(onCategorySelected: _handleCategorySelected),
              ContentSection(contentController: _content, formKey: _contentKey)
          ]),
          bottomButtons(
            MediaQuery.of(context).size.width, 
            outlinedRoundedRectangleButton('AI로 생성하기', () {
              checkValidCategory(_selectedCategory);
              
              if (_titleKey.currentState!.validate()) {
                createScriptByGpt(_title.text, _selectedCategory!);
              }
            }), 
            fullyRoundedRectangleButton(colors.buttonColor, '완료', () { 
              checkValidCategory(_selectedCategory);

              if (_titleKey.currentState!.validate() & _contentKey.currentState!.validate()) {
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