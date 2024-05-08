import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:capstone/constants/text.dart' as texts;

class GptController {
  static final GptController _instance = GptController._internal();

  factory GptController() {
    return _instance;
  }

  GptController._internal();

  Future<String> createScript(String title, String category) async {
    var fastApiUrl = Uri.parse("${texts.baseUrl}/script");
    http.Response response = await http.post(
      fastApiUrl,
      headers: {
        'Content-Type': 'application/json'
      },
      body: jsonEncode({
        "title": title,
        "category": category
      }),  
    );

    final responseData = jsonDecode(utf8.decode(response.bodyBytes));
    
    return responseData['script'];
  }
}