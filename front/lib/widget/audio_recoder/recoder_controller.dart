import 'dart:async';
import 'package:get/get.dart';
import 'package:record/record.dart';

class RecorderController extends GetxController {
  late AudioRecorder _audioRecorder;
  final Rx<RecordState?> _recordState = Rx<RecordState?>(RecordState.stop);
  final Rx<Amplitude?> _amplitude = Rx<Amplitude?>(null);
  StreamSubscription<RecordState>? _stateSub;
  StreamSubscription<Amplitude>? _amplitudeSub;

  @override
  void onInit() {
    _audioRecorder = AudioRecorder();
    _stateSub = _audioRecorder.onStateChanged().listen((state) {
      _recordState.value = state;
    });

    // Replace with your desired duration
    const duration = Duration(seconds: 1);
    _amplitudeSub = _audioRecorder.onAmplitudeChanged(duration).listen((amp) {
      _amplitude.value = amp;
    });

    // 원하는 녹음 품질과 옵션 설정
    const recordConfig = RecordConfig(
        sampleRate: 44100, // 샘플 레이트: 44100 Hz (기본값)
        bitRate: 128000, // 비트 레이트: 128000 bps (기본값)
        noiseSuppress: true);

    super.onInit();
  }

  void setAmplitude(Duration duration) {
    _amplitudeSub?.cancel();
    _amplitudeSub = _audioRecorder.onAmplitudeChanged(duration).listen((amp) {
      _amplitude.value = amp;
    });
  }

  void startRecording({required String filePath}) async {
    await _audioRecorder.start(path: filePath);
  }

  void stopRecording() async {
    await _audioRecorder.stop();
  }

  void cancelRecording() async {
    await _audioRecorder.cancel();
  }

  @override
  void onClose() {
    _stateSub?.cancel();
    _amplitudeSub?.cancel();
    _audioRecorder.dispose();
    super.onClose();
  }
}

class RecorderState {
  const RecorderState();
}
