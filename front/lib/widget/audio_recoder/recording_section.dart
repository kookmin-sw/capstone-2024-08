import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:capstone/widget/audio_player.dart';
import 'package:capstone/widget/audio_recoder/audio_recorder.dart';

class RecordingSection extends StatefulWidget {
  final bool showPlayer;
  final String? audioPath;
  final void Function(bool isShowPlayer, String? path) onDone;

  const RecordingSection({
    Key? key,
    required this.showPlayer,
    required this.audioPath,
    required this.onDone,
  }) : super(key: key);

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
    return Container(
      width: MediaQuery.of(context).size.width / 1.2,
      padding: EdgeInsets.all(20),
      child: showPlayer
          ? AudioPlayer(
              source: audioPath!,
              onDelete: () {
                setState(() {
                  showPlayer = false;
                  audioPath = null;
                  widget.onDone(showPlayer, audioPath);
                });
              },
            )
          : Recorder(
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
