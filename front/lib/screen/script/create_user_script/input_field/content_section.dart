import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

class ContentSection extends StatefulWidget {
  const ContentSection({
      super.key, 
      required this.contentController, 
      required this.formKey, 
  });

  final TextEditingController contentController;
  final GlobalKey<FormState> formKey;

  @override
  State<ContentSection> createState() => _ContentSectionState();
}

class _ContentSectionState extends State<ContentSection> {
  @override
  Widget build(BuildContext context) {
    return Form(
      key: widget.formKey,
      child: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height * 0.4,
        margin: const EdgeInsets.fromLTRB(30, 10, 30, 30),
        decoration: BoxDecoration(
          color: colors.blockColor,
          borderRadius: BorderRadius.circular(10),
          boxShadow: const [
            BoxShadow(
              color: colors.buttonSideColor,
              blurRadius: 2,
              spreadRadius: 2,
            )
          ]),
        padding: const EdgeInsets.all(15),
        child: TextFormField(
            autofocus: true,
            maxLines: null,
            keyboardType: TextInputType.multiline,
            textInputAction: TextInputAction.next,
            validator: (value) {
              if (value == null || value.isEmpty) {
                return '내용은 비어있을 수 없습니다';
              }
              return null;
            },
            decoration: const InputDecoration(
                labelText: '내용을 직접 입력하거나 AI로 생성해보세요.',
                labelStyle: TextStyle(
                  color: colors.textColor,
                  fontWeight: FontWeight.w500
                ),
                border: InputBorder.none
                ),
            controller: widget.contentController,
          )
      ));
  }
}
