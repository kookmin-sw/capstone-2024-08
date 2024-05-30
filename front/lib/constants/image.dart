import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:capstone/constants/color.dart' as colors;

final loroLogo = Image.asset(
  'assets/images/authentication/loro_logo.png',
  semanticLabel: 'Loro logo',
);

final googleIcon = Image.asset(
  'assets/images/authentication/google_icon.png',
  semanticLabel: 'Google 아이콘',
);

final tosLine = SvgPicture.asset(
  'assets/images/authentication/tos_line.svg',
  color: colors.blockColor,
  semanticsLabel: '이용약관 라인',
);

final policyLine = SvgPicture.asset(
  'assets/images/authentication/policy_line.svg',
  color: colors.blockColor,
  semanticsLabel: '개인정보 처리방침 라인',
);

final Map<String, String> characterImagePaths = {
  'panda': 'assets/images/characters/panda.png',
  'fox': 'assets/images/characters/fox.png',
  'frog': 'assets/images/characters/frog.png',
  'hippo': 'assets/images/characters/hippo.png',
};

List<String?> characterForSetup = [
  characterImagePaths['panda'],
  characterImagePaths['fox'],
  characterImagePaths['frog'],
  characterImagePaths['hippo'],
];