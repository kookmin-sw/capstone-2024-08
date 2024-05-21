import 'package:capstone/widget/practice/prompt/prompt_guide_player.dart';
import 'package:capstone/widget/practice/prompt/prompt_recoder.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class PromptRecordingSection extends StatefulWidget {
  final bool showPlayer;
  final String? audioPath;
  final void Function(bool isShowPlayer, String? path) onDone;

  const PromptRecordingSection({
    super.key,
    required this.showPlayer,
    required this.audioPath,
    required this.onDone,
  });

  @override
  State<PromptRecordingSection> createState() => _PromptRecordingSectionState();
}

class _PromptRecordingSectionState extends State<PromptRecordingSection> {
  bool showPlayer = false;
  String? audioPath;

  @override
  void initState() {
    showPlayer = false;
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: MediaQuery.of(context).size.width / 1.2,
      padding: EdgeInsets.all(20),
      child: showPlayer
          ? GuideVoicePlayer(
              source: audioPath!,
              onStop: () {},
              onDelete: () {},
            )
          : PromptRecoder(
              onStop: (path) {
                if (kDebugMode) print('Recorded file path: $path');
                setState(() {
                  audioPath = path;
                  showPlayer = true;
                  widget.onDone(showPlayer, audioPath);
                });
              },
            ),
    );
  }
}
