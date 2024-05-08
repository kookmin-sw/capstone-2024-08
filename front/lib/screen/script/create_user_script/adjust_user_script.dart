import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/model/save_data.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/script/create_user_script/controller/content_controller.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/bottom_buttons.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/outlined_rounded_rectangle_button.dart';
import 'package:capstone/widget/script/script_content_adjust_block.dart';
import 'package:capstone/screen/script/select_practice.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';


class AdjustUserScript extends StatefulWidget {
  const AdjustUserScript({
    Key? key,
    required this.title,
    required this.category,
  }) : super(key: key);

  final String title;
  final String category;

  @override
  State<AdjustUserScript> createState() => _AdjustUserScriptState();
}

class _AdjustUserScriptState extends State<AdjustUserScript> {
  SaveData saveData = SaveData();

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
 
  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;

    return Scaffold(
      appBar: basicAppBar(title: '나만의 대본 만들기'),
      body: Stack(
              children: [
                Container(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
                  child: ListView(
                    children: [
                      _buildCategory(widget.category),
                      const SizedBox(height: 15),
                      _buildTitle(widget.title),
                      const SizedBox(height: 20),
                      GetBuilder<UserScriptContentController>(
                        builder: (controller){
                          return scriptContentAdjustBlock(controller, width);
                        }
                      ),
                      const SizedBox(height: 30),
                  ])
                ),
                bottomButtons(
                  width, 
                  outlinedRoundedRectangleButton('저장 후 나가기', () {
                      saveUserScript();                    
                      Get.close(2);
                  }), 
                  fullyRoundedRectangleButton(colors.buttonColor, '연습하기', () {
                      ScriptModel userScript = saveUserScript();
                      Get.to(() => SelectPractice(
                        script: userScript,
                        tapCloseButton: () { Get.close(3); },
                      ));
                  })
              )]
      ));
  }

  ScriptModel saveUserScript(){
    List<TextEditingController> controllers = Get.find<UserScriptContentController>().textEditingControllerList!;
    List<String> sentenceList = [
      for(TextEditingController controller in controllers)
        controller.text
    ];
    ScriptModel userScript = ScriptModel(
        //uid: user!.uid,
        title: widget.title,
        category: widget.category,
        content: sentenceList,
        createdAt: Timestamp.now(),
    );

    saveData.addUserScript('mg', userScript);

    Get.delete<UserScriptContentController>();

    return userScript;
  }
}