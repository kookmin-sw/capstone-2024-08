import 'package:capstone/model/script.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:capstone/widget/practice/scrap_button.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:capstone/constants/color.dart' as colors;

class PracticeAppBar extends StatefulWidget implements PreferredSizeWidget {
  bool backButton = true;
  List<int> scrapSentences = [];
  final ScriptModel script;
  final String scriptType;
  int sentenceIndex;

  PracticeAppBar(
      {Key? key,
      required this.script,
      required this.scriptType,
      required this.backButton,
      required this.scrapSentences,
      required this.sentenceIndex})
      : super(key: key);

  @override
  State<PracticeAppBar> createState() => _PracticeAppBarState();

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}

class _PracticeAppBarState extends State<PracticeAppBar> {
  String uid = Get.find<UserController>().userModel.id!;
  List<int> scrapSentences = [];

  @override
  void initState() {
    super.initState();
    scrapSentences = widget.scrapSentences;
  }

  bool isClicked(int sentenceIndex) {
    if (scrapSentences.contains(sentenceIndex)) {
      return true;
    }
    return false;
  }

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: colors.bgrBrightColor,
      elevation: 0,
      leading: widget.backButton
          ? IconButton(
              icon: const Icon(Icons.keyboard_backspace_rounded,
                  color: colors.bgrBrightColor),
              onPressed: () => Get.back())
          : null,
      actions: [
        scrapsButton(widget.scriptType, widget.script.id!, uid,
            widget.sentenceIndex, isClicked(widget.sentenceIndex))
      ],
    );
  }
}
