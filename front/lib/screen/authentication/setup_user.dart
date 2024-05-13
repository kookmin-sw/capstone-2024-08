import 'package:capstone/model/user.dart';
import 'package:capstone/screen/authentication/controller/auth_controller.dart';
import 'package:capstone/screen/bottom_navigation.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/screen/authentication/get_user_voice.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/image.dart' as images;
import 'package:get/get.dart';

class SetupUser extends StatefulWidget {
  const SetupUser({Key? key}) : super(key: key);

  @override
  State<SetupUser> createState() => _SetupUserState();
}

class _SetupUserState extends State<SetupUser> {
  final TextEditingController _nickname = TextEditingController();
  String? _selectedCharacter = images.characterForSetup[0];
  final _formKey = GlobalKey<FormState>();
  final User? user = FirebaseAuth.instance.currentUser;

  void _handleCharacterSelected(String character) {
    setState(() {
      _selectedCharacter = character;
    });
  }

  Widget _nicknameSection() {
    return Form(
        key: _formKey,
        child: Container(
            width: getDeviceWidth(context) * 0.9,
            decoration: _boxDecoration(28),
            child: TextFormField(
              maxLength: 8,
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return '닉네임은 비어있을 수 없습니다';
                }
                return null;
              },
              decoration: InputDecoration(
                labelText: '닉네임을 입력해주세요.',
                labelStyle: const TextStyle(
                    color: colors.textColor, fontWeight: FontWeight.w500),
                floatingLabelBehavior: FloatingLabelBehavior.never,
                fillColor: colors.blockColor,
                filled: true,
                counterText: '',
                border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(28),
                    borderSide: BorderSide.none),
              ),
              controller: _nickname,
            )));
  }

  BoxDecoration _boxDecoration(double radius) {
    return BoxDecoration(
        color: colors.blockColor,
        borderRadius: BorderRadius.circular(radius),
        boxShadow: const [
          BoxShadow(
            color: colors.buttonSideColor,
            blurRadius: 3,
            spreadRadius: 3,
          )
        ]);
  }

  Widget _buildSelectedCharacter() {
    return Container(
      width: getDeviceWidth(context) * 0.5,
      height: getDeviceHeight(context) * 0.2,
      decoration: _boxDecoration(10),
      child: Image.asset(_selectedCharacter!),
    );
  }

  Widget _buildCharacterList() {
    return Container(
        width: getDeviceWidth(context) * 0.9,
        height: getDeviceHeight(context) * 0.4,
        decoration: _boxDecoration(10),
        child:
            Column(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
          for (int paragraph = 0; paragraph < 2; paragraph++)
            Row(mainAxisAlignment: MainAxisAlignment.spaceAround, children: [
              for (int idx = 2 * paragraph; idx < 2 * paragraph + 2; idx++)
                IconButton(
                  icon: Image.asset(
                    images.characterForSetup[idx]!,
                    width: getDeviceWidth(context) * 0.3,
                  ),
                  onPressed: () =>
                      _handleCharacterSelected(images.characterForSetup[idx]!),
                )
            ])
        ]));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: basicAppBar(title: '회원가입', backButton: false),
      body: GestureDetector(
        onTap: () {
          FocusScope.of(context).unfocus();
        },
        child: ListView(
          children: [
            Center(
              child: Column(
                children: [
                  const SizedBox(height: 20),
                  _nicknameSection(),
                  const SizedBox(height: 20),
                  _buildSelectedCharacter(),
                  const SizedBox(height: 20),
                  _buildCharacterList(),
                  const SizedBox(height: 20),
                  Container(
                    width: getDeviceWidth(context) * 0.9,
                    child:
                        fullyRoundedRectangleButton(colors.blockColor, '완료', () async {
                      if (_formKey.currentState!.validate()) {
                        UserModel userData = UserModel(
                            id: user!.uid,
                            nickname: _nickname.text,
                            character: _selectedCharacter!.split('/')[3].split('.')[0],
                            attendanceStreak: null,
                            lastAccessDate: null,
                            lastPracticeScript: null,
                            voiceUrls: {'long': "", 'middle': "", 'short': ""});
                        Get.to(() => GetUserVoice(userData: userData));
                      }
                    }))
                ])
            )
        ]),
    ));
  }
}
