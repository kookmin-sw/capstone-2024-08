import 'dart:io';
import 'dart:typed_data';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

mixin AudioRecorderMixin {
  Future<void> recordFile(
    AudioRecorder recorder,
    RecordConfig config,
  ) async {
    final path = await _getPath();

    try {
      await recorder.start(config, path: path);
    } catch (e) {
      print('Error starting recorder: $e');
    }
  }

  Future<void> recordStream(AudioRecorder recorder, RecordConfig config) async {
    final path = await _getPath();
    final file = File(path);

    try {
      final stream = await recorder.startStream(config);

      stream.listen(
        (data) {
          try {
            print(recorder.convertBytesToInt16(Uint8List.fromList(data)));
            file.writeAsBytesSync(data, mode: FileMode.append);
          } catch (e) {
            print('Error writing to file: $e');
          }
        },
        onDone: () {
          print('End of stream. File written to $path.');
        },
        onError: (error) {
          print('Stream error: $error');
        },
      );
    } catch (e) {
      print('Error starting stream: $e');
    }
  }

  void downloadWebData(String path) {
    // Implement your web data download logic here
  }

  Future<String> _getPath() async {
    final dir = await getTemporaryDirectory();
    return p.join(
      dir.path,
      'audio_${DateTime.now().millisecondsSinceEpoch}.wav',
    );
  }
}
