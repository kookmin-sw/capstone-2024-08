import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

Positioned bottomButtons(var width, Widget leftButton, Widget rightButton) {
  return Positioned(
      left: 0,
      right: 0,
      bottom: 0,
      child: Container(
          padding: const EdgeInsets.all(5),
          decoration: const BoxDecoration(
            color: colors.blockColor, 
            boxShadow: [
              BoxShadow(
                color: colors.buttonSideColor,
                blurRadius: 5,
                spreadRadius: 5,
              )
          ]),
          child: Container(
              padding: const EdgeInsets.fromLTRB(20, 5, 20, 5),
              child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Container(
                        width: width * 0.4,
                        child: leftButton
                    ),
                    Container(
                      width: width * 0.4,
                      child: rightButton 
                    ),
                  ])
          )
      )
  );
}