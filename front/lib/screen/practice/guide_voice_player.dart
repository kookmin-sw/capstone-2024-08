import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:flutter/widgets.dart';
import 'dart:io';
import 'package:capstone/constants/text.dart' as texts;
import 'package:get/get.dart';
import 'package:http/http.dart' as http;

class GuideVoicePlayer extends StatefulWidget {
  GuideVoicePlayer({Key? key, required this.text}) : super(key: key);
  final String text;

  @override
  State<GuideVoicePlayer> createState() => _GuideVoicePlayerState();
}

class _GuideVoicePlayerState extends State<GuideVoicePlayer> {
  Map<String, File?> _wavFiles = Get.find<UserController>().wavFiles;

  @override
  void initState() {
    super.initState();
  }

  Future<void> sendDataToServer(
      String text, Map<String, File?> wavFiles) async {
    var url = Uri.parse('${texts.baseUrl}/feedback/');

    // 멀티파트 리퀘스트 생성
    var request = http.MultipartRequest('POST', url);

    // 텍스트 필드 추가
    request.fields['text'] = widget.text;

    // WAV 파일들 추가
    for (var entry in wavFiles.entries) {
      var wavFile = entry.value;
      if (wavFile != null) {
        var wavStream = http.ByteStream(wavFile.openRead());
        var length = await wavFile.length();
        var multipartFile = http.MultipartFile(
          'wavs',
          wavStream,
          length,
          filename: '${entry.key}.wav', // 파일 이름 지정
        );
        request.files.add(multipartFile);
      }
    }

    // 리퀘스트 보내기
    var response = await request.send();

    // 응답 확인
    if (response.statusCode == 200) {
      print('데이터가 성공적으로 서버로 전송되었습니다.');
    } else {
      print('데이터 전송에 실패했습니다. 상태 코드: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return const Placeholder();
  }
}
