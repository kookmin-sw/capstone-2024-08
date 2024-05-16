import 'dart:convert';

import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:flutter/widgets.dart';
import 'dart:io';
import 'package:capstone/constants/text.dart' as texts;
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

Future<String?> sendDataToServerAndDownLoadGuideVoice(
    String text, Map<String, File?> wavFiles) async {
  var url = Uri.parse('${texts.baseUrl}/voice_guide/');

  print("함수 내에서의 사용자 음성 파일들 : $wavFiles");

  // 멀티파트 리퀘스트 생성
  var request = http.MultipartRequest('POST', url);

  // 텍스트 필드 추가
  request.fields['sentence'] = text;

  // WAV 파일들 추가
  for (var entry in wavFiles.entries) {
    var wavFile = entry.value;
    print("Wav File Path : ${wavFile!.path}");
    if (wavFile.existsSync()) {
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

  print('request: ${request}');

  // 리퀘스트 보내기
  var response = await request.send();

  // 응답 확인
  if (response.statusCode == 200) {
    // 서버에서 받은 응답 파싱
    var responseData = jsonDecode(await response.stream.bytesToString());

    // 응답 데이터에서 WAV 파일의 URL 가져오기
    var wavUrl = responseData['wav_url'];
    print(wavUrl);

    // WAV 파일 다운로드
    var wavResponse = await http.get(Uri.parse(wavUrl));

    // 다운로드한 WAV 파일을 로컬에 저장
    if (wavResponse.statusCode == 200) {
      var bytes = wavResponse.bodyBytes;
      Directory dir = await getTemporaryDirectory();
      String localPath =
          '${dir.path}/guide_voices_${DateTime.now().millisecondsSinceEpoch}.wav';
      var file = File(localPath); // 파일 저장 경로 지정
      try {
        await file.writeAsBytes(bytes);
        print('WAV 파일이 성공적으로 저장되었습니다.');
        return localPath;
      } catch (e) {
        print('WAV 파일 저장에 실패했습니다: $e');
      }
    } else {
      print('WAV 파일 다운로드에 실패했습니다. 상태 코드: ${wavResponse.statusCode}');
    }
  } else {
    print('텍스트 전송에 실패했습니다. 상태 코드: ${response.statusCode}');
  }
  return null;
}

Future<int?> getVoicesSimilarity(
    String text, String currentSentencePracticeWavFilePath) async {
  try {
    var url = Uri.parse('${texts.baseUrl}/feedback/');

    // 멀티파트 리퀘스트 생성
    var request = http.MultipartRequest('POST', url);

    // 텍스트 필드 추가
    request.fields['sentence'] = text;

    // WAV 파일 추가
    var wavFile = File(currentSentencePracticeWavFilePath);
    if (!wavFile.existsSync()) {
      print('파일이 존재하지 않습니다.');
      return null;
    }

    var wavStream = http.ByteStream(wavFile.openRead());
    var length = await wavFile.length();
    var multipartFile = http.MultipartFile(
      'user_wav',
      wavStream,
      length,
      filename: 'currentSentencePracticeWavFile.wav', // 파일 이름 지정
    );
    request.files.add(multipartFile);

    // 리퀘스트 보내기
    var response = await request.send();

    // 응답 확인
    if (response.statusCode == 200) {
      // 서버에서 받은 응답 파싱
      var responseData = await response.stream.bytesToString();
      int precision =
          jsonDecode(responseData)['similarity_percentage'].truncate();

      print(responseData);

      // 응답 데이터를 정수로 변환하여 반환
      return precision;
    } else {
      print('서버 요청에 실패했습니다. 상태 코드: ${response.statusCode}');
      return null;
    }
  } catch (e) {
    print('오류 발생: $e');
    return null;
  }
}
