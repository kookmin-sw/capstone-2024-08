import 'package:capstone/screen/search/search_taps.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/fonts.dart' as fonts;
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
        padding: EdgeInsets.only(right: getDeviceWidth(context) * 0.075),
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
            labelStyle: TextStyle(
              color: colors.textColor,
              fontSize: fonts.plainText(context),
              fontWeight: FontWeight.w500),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(28),
              borderSide: BorderSide.none),
          ))),
    );
  }

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);
    
    return Scaffold(
      body: Column(
          children: [
            Container(
              color: colors.bgrDarkColor,
              padding: EdgeInsets.fromLTRB(deviceWidth * 0.04, deviceWidth * 0.07, deviceWidth * 0.03, 0),
              width: deviceWidth,
              child: Row(
                children: [
                  backToPreviousPage(),
                  keywordSection()
              ])
            ),
            Flexible(
              child: GestureDetector(
                onTap: () {
                  FocusScope.of(context).unfocus();
                },
                child: SearchTabs(query: query.text),
              )
            )
        ])
    );
  }
}
