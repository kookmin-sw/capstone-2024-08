import 'package:capstone/screen/bottom_navigation.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/fully_rounded_rectangle_button.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/auth_icons.dart' as auth_icons;
import 'package:get/get.dart';

class SetupUser extends StatefulWidget {
  const SetupUser({Key? key}) : super(key: key);

  @override
  State<SetupUser> createState() => _SetupUserState();
}

class _SetupUserState extends State<SetupUser> {
  final TextEditingController _nickname = TextEditingController();
  Image? _selectedCharacter = auth_icons.characters[0];
  final _formKey = GlobalKey<FormState>();
  final User? user = FirebaseAuth.instance.currentUser;

  void _handleCharacterSelected(Image character) {
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
          maxLines: 1,
          textInputAction: TextInputAction.next,
          validator: (value) {
            if (value == null || value.isEmpty) {
              return '닉네임은 비어있을 수 없습니다';
            }
            return null;
          },
          decoration: InputDecoration(
            labelText: '닉네임을 입력해주세요.',
              labelStyle: const TextStyle(
                color: colors.textColor,
                fontWeight: FontWeight.w500
              ),
            floatingLabelBehavior: FloatingLabelBehavior.never,
            fillColor: colors.blockColor,
            filled: true,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(28),
              borderSide: BorderSide.none),
          ),
          controller: _nickname,
        )
      )
    );
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
      ]
    );
  }

  Widget _buildSelectedCharacter() {
    return Container(
      width: getDeviceWidth(context) * 0.5,
      height: getDeviceHeight(context) * 0.2,
      decoration: _boxDecoration(10),
      child: _selectedCharacter, 
    );
  }

  Widget _buildCharacterList() {
    return Container(
      width: getDeviceWidth(context) * 0.9,
      height: getDeviceHeight(context) * 0.4,
      decoration: _boxDecoration(10),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceAround, 
        children: [
          for (int paragraph = 0; paragraph < 2; paragraph++)
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround, 
              children: [
                for (int idx = 2 * paragraph; idx < 2 * paragraph + 2; idx++)
                  IconButton(
                    icon: auth_icons.characters[idx],
                    iconSize: getDeviceWidth(context) * 0.3,
                    onPressed: () => _handleCharacterSelected(auth_icons.characters[idx]),
                  )
          ])
      ]) 
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: basicAppBar(title: '회원가입', backButton: false),
      body: SingleChildScrollView(
        child: Center(
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
                child: fullyRoundedRectangleButton(colors.blockColor, '완료', () {
                  if (_formKey.currentState!.validate()) {
                    FirebaseFirestore.instance
                    .collection('user')
                    .doc(user!.uid)
                    .set({
                      'nickname': _nickname.text,
                      'character': "unicorn"
                    });
                    Get.to(() => const BottomNavBar());
                  }
                })
              )
          ])
        )
      ),
    );
  }
}
