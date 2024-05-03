import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

class TitleSection extends StatefulWidget {
  const TitleSection({
    super.key, 
    required this.titleController,
    required this.formKey, 
  });

  final TextEditingController titleController;
  final GlobalKey<FormState> formKey;

  @override
  State<TitleSection> createState() => _TitleSectionState();
}

class _TitleSectionState extends State<TitleSection> {
  @override
  Widget build(BuildContext context) {
    return Form(
      key: widget.formKey,
        child: Container(
        margin: const EdgeInsets.fromLTRB(30, 30, 30, 10),
        width: MediaQuery.of(context).size.width,
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
                return '제목은 비어있을 수 없습니다';
              }
              return null;
            },
            decoration: const InputDecoration(
                labelText: '제목을 입력하세요.',
                labelStyle: TextStyle(
                  color: colors.textColor,
                  fontWeight: FontWeight.w500
                ),
                border: InputBorder.none
                ),
            controller: widget.titleController,
          )
      ));
  }
}
