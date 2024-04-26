import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/screen/setting/policy.dart';

List<Map<String, dynamic>> settingItems = [
  {
    'name': '사용자 음성정보 재설정', 
    'route': Policy(policy: texts.personalData)
  },
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
    'route': Policy(policy: texts.personalData)
  },
  {
    'name': '탈퇴하기',
    'route': Policy(policy: texts.personalData)
  }
];