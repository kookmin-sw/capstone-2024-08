import 'package:capstone/screen/search/search_taps.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:get/get.dart';

class SearchScript extends StatefulWidget {
  const SearchScript({Key? key}) : super(key: key);

  @override
  State<SearchScript> createState() => _SearchScriptState();
}

class _SearchScriptState extends State<SearchScript> {
  final TextEditingController query = TextEditingController();

  IconButton backToPreviousPage() {
    return IconButton(
      icon: const Icon(
        Icons.arrow_back_rounded, 
        color: colors.blockColor
      ),
      onPressed: () => Get.back(),
    );
  }

  Flexible keywordSection() {
    return Flexible(
      child: Container(
        padding: const EdgeInsets.fromLTRB(0, 0, 20, 0),
        child: TextFormField(
          onChanged: (text) {
            setState(() {});
          },
          controller: query,
          maxLines: 1,
          decoration: InputDecoration(
            labelText: '검색할 키워드를 입력해주세요.',
            floatingLabelBehavior: FloatingLabelBehavior.never,
            fillColor: colors.blockColor,
            filled: true,
            labelStyle: const TextStyle(
              color: colors.textColor,
              fontSize: 12,
              fontWeight: FontWeight.w500),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(28),
              borderSide: BorderSide.none),
          ))),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        color: colors.bgrDarkColor,
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
              width: MediaQuery.of(context).size.width,
              child: Row(
                children: [
                  backToPreviousPage(),
                  keywordSection()
              ])
            ),
            Flexible(
              child: SearchTabs(query: query.text),
            )
        ])
    ));
  }
}
