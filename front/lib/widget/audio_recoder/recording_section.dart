import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:capstone/widget/audio_player.dart';
import 'package:capstone/widget/audio_recoder/audio_recorder.dart';

class RecordingSection extends StatefulWidget {
  const RecordingSection({super.key});

  @override
  State<RecordingSection> createState() => _RecordingSectionState();
}

class _RecordingSectionState extends State<RecordingSection> {
  bool showPlayer = false;
  String? audioPath;

  @override
  void initState() {
    showPlayer = false;
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Center(
      child: showPlayer
          ? Padding(
              padding: const EdgeInsets.symmetric(horizontal: 25),
              child: AudioPlayer(
                source: audioPath!,
                onDelete: () {
                  setState(() => showPlayer = false);
                },
              ),
            )
          : Recorder(
              onStop: (path) {
                if (kDebugMode) print('Recorded file path: $path');
                setState(() {
                  audioPath = path;
                  showPlayer = true;
                });
              },
            ),
    ));
  }
}
