import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/model/load_data.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/screen/setting/setting.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/widget/home/attendance_streak.dart';
import 'package:capstone/widget/home/character_section.dart';
import 'package:capstone/widget/script/script_list_tile.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  // final User? user = FirebaseAuth.instance.currentUser;
  final LoadData loadData = LoadData();

  String? nickname = Get.find<UserController>().userModel.nickname;
  String? character = Get.find<UserController>().userModel.character;
  int? attendanceStreakDays =
      Get.find<UserController>().userModel.attendanceStreak;
  DocumentReference? lastPracticeScript =
      Get.find<UserController>().userModel.lastPracticeScript;

  AppBar homeAppBar() {
    return AppBar(
      backgroundColor: colors.bgrDarkColor,
      // text overflow 반영 필요 : libary로 화면 비율에 따른 조정 반영 필요
      title: Text("${texts.homeWelcomeMessage} $nickname님 !",
          style: const TextStyle(
              color: colors.themeWhiteColor, fontWeight: FontWeight.bold)),
      actions: [
        IconButton(
          icon:
              const Icon(Icons.settings_rounded, color: colors.themeWhiteColor),
          onPressed: () {
            HapticFeedback.lightImpact();
            Get.to(() => const Setting());
          },
        )
      ],
    );
  }

  Widget _characterSection(String? character) {
    if (character == null)
      return Text('캐릭터 없음');
    else {
      return Container(
          width: MediaQuery.of(context).size.width / 1.7,
          child: characterSection(character));
    }
  }

  Widget _attendanceStreak(String? ninkname, int? attendanceStreakDays) {
    return Container(
        child: attendanceStreakSection(nickname, attendanceStreakDays));
  }

  Widget _lastPracticeScript(
      BuildContext context, DocumentReference? lastPracticeScript) {
    if (lastPracticeScript == null) {
      return Text(
        texts.lastPracticeScriptEmptyMessage,
        style: const TextStyle(color: colors.themeWhiteColor),
      );
    } else {
      return Container();
      // return FutureBuilder<ScriptModel?>(
      //   future: loadData.readScriptByDocumentRef(lastPracticeScript),
      //   builder: (context, snapshot) {
      //     if (snapshot.connectionState == ConnectionState.waiting) {
      //       return const CircularProgressIndicator(
      //         color: colors.exampleScriptColor,
      //       ); // 로딩 중이면 로딩 인디케이터 표시
      //     } else if (snapshot.hasError) {
      //       return Text('Error: ${snapshot.error}'); // 에러가 있으면 에러 메시지 표시
      //     } else {
      //       // 데이터가 로드되면 스크립트를 표시
      //       final script = snapshot.data;
      //       if (script != null) {
      //         return scriptListTile(context, script);
      //       } else {
      //         return Text('No script found'); // 스크립트가 없으면 해당 메시지 표시
      //       }
      //     }
      //   },
      // );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: colors.bgrDarkColor,
        appBar: homeAppBar(),
        body: Column(children: [
          _characterSection(character),
          _lastPracticeScript(context, lastPracticeScript),
          _attendanceStreak(nickname, attendanceStreakDays)
        ]));
  }
}
