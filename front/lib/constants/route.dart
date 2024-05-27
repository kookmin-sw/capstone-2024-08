import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/screen/setting/policy.dart';
import 'package:capstone/screen/setting/setting.dart';

List<Map<String, dynamic>> settingItems = [
  {
    'name': '이용약관', 
    'route': Policy(policy: texts.usingPolicy)
  },
  {
    'name': '개인정보처리방침', 
    'route': Policy(policy: texts.personalData)
  },
  {
    'name': '로그아웃',
    'action': handleLogoutAction,
  },
  {
    'name': '탈퇴하기',
    'action': handleDeleteAction,
  }
];