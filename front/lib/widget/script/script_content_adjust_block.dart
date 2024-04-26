import 'package:capstone/screen/script/create_user_script/content_controller.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

Column scriptContentAdjustBlock(UserScriptContentController controller, var width) {
  return Column(
    crossAxisAlignment: CrossAxisAlignment.center,
    mainAxisAlignment: MainAxisAlignment.start,
    children: [
      Column(
        children: [
          for(int idx=0; idx<controller.textEditingControllerList!.length; idx++) 
                Container(
                  width: width,
                  margin: const EdgeInsets.only(bottom: 20),
                  padding: const EdgeInsets.fromLTRB(20, 5, 15, 5),
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
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: TextFormField(
                        maxLines: null,
                        keyboardType: TextInputType.multiline,
                        controller: controller.textEditingControllerList![idx],
                        decoration: const InputDecoration(
                          border: InputBorder.none
                        ),
                      )),
                      IconButton(
                        icon: const Icon(Icons.remove_circle_outline_rounded, color: colors.bgrDarkColor),
                        onPressed: () {
                          controller.removeController(idx);
                        },
                      )
              ])
            )
      ]),
      IconButton(
        icon: const Icon(Icons.add_circle_outline_rounded, color: colors.bgrDarkColor),
        onPressed: () {
          controller.addController();
        },
      )
  ]);
}