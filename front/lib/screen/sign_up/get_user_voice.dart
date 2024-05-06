import 'package:capstone/screen/sign_up/voice_collection.dart';
import 'package:flutter/material.dart';

class VoiceRecorderApp extends StatelessWidget {
  const VoiceRecorderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Voice Recorder'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('버튼을 눌러보세요'),
            IconButton(
                onPressed: () {
                  showModalBottomSheet<String>(
                      context: context, builder: (context) => RecordingSheet());
                },
                icon: const Icon(Icons.graphic_eq)),
          ],
        ),
      ),
    );
  }
}
