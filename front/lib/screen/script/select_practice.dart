import 'package:capstone/model/script.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:flutter/services.dart';

class SelectPractice extends StatelessWidget {
  const SelectPractice({
    required this.script,
    required this.tapCloseButton,
    super.key,
  });

  final ScriptModel script;
  final Function tapCloseButton;
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        color: colors.selectPracticebgrColor,
        child: Column(
          children: [
            Align(
              alignment: Alignment.topRight,
              child: IconButton(
                icon: const Icon(Icons.close_rounded, color: colors.blockColor, size: 40),
                onPressed: () { tapCloseButton(); }, 
              )
            ),
            SizedBox(height: MediaQuery.of(context).size.height * 0.3),
            Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                notice(),
                const SizedBox(height: 25),
                practiceButton(context, '프롬프트', () {}),
                const SizedBox(height: 25),
                practiceButton(context, '문장단위연습', () {})  
              ])
        ])
    ));
  }
}

Text notice(){
  return const Text(
    '연습 방법을 선택해주세요.',
    style: TextStyle(
      color: colors.blockColor,
      fontSize: 13,
      fontWeight: FontWeight.w500,     
    )    
  );
}

Container practiceButton(BuildContext context, String buttonText, Function pressedFunc) {
  return Container(
    width: MediaQuery.of(context).size.width * 0.8,
    height: MediaQuery.of(context).size.height * 0.08,
    child: ElevatedButton(
      style: ElevatedButton.styleFrom(
          backgroundColor: colors.blockColor,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
      onPressed: () {
        HapticFeedback.lightImpact();
        pressedFunc();
      },
      child: Text(
        buttonText,
        semanticsLabel: buttonText,
        textAlign: TextAlign.center,
        style: const TextStyle(
          color: colors.textColor,
          fontSize: 13,
          fontWeight: FontWeight.w800,
        ),
      )
    ));
}