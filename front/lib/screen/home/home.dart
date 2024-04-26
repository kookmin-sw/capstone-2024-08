import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/screen/setting/setting.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  String nickname = "로로";

  AppBar homeAppBar() {
    return AppBar(
      // text overflow 반영 필요 : libary로 화면 비율에 따른 조정 반영 필요
      title: Text(texts.homeWelcomeMessage + "$nickname님 !",
          style: TextStyle(color: Colors.white)),
      actions: [
        IconButton(
          icon: const Icon(Icons.settings, color: Colors.white),
          onPressed: () {
            HapticFeedback.lightImpact();
            Get.to(() => const Setting());
          },
        )
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: colors.bgrDarkColor,
        appBar: homeAppBar(),
        body: Column(children: []));
  }
}
