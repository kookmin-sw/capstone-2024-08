import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:capstone/constants/color.dart' as colors;

final loroLogo = Image.asset(
  'assets/icons/authentication/login/loro_logo.png',
  semanticLabel: 'Loro logo',
);

final googleIcon = Image.asset(
  'assets/icons/authentication/login/google_icon.png',
  semanticLabel: 'Google 아이콘',
);

final tosLine = SvgPicture.asset(
  'assets/icons/authentication/login/tos_line.svg',
  color: colors.blockColor,
  semanticsLabel: '이용약관 라인',
);

final policyLine = SvgPicture.asset(
  'assets/icons/authentication/login/policy_line.svg',
  color: colors.blockColor,
  semanticsLabel: '개인정보 처리방침 라인',
);

List<Image> characters = [googleIcon, googleIcon, googleIcon, googleIcon];