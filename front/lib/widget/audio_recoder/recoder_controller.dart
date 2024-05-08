import 'dart:async';
import 'dart:math';
import 'package:get/get.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';

class RecorderController extends GetxController {
  late AudioRecorder _audioRecorder;
  final Rx<RecordState?> _recordState = Rx<RecordState?>(RecordState.stop);
  final Rx<Amplitude?> _amplitude = Rx<Amplitude?>(null);
  StreamSubscription<RecordState>? _stateSub;
  StreamSubscription<Amplitude>? _amplitudeSub;
  late RecordConfig recordConfig;
  static const double maxAmplitude = 100.0;
  Rx<Map<String, double>> ampl =
      Rx<Map<String, double>>({'max': 0.0, 'current': 0.0});

  void updateAmpl({required double intiMax, required double initCurrent}) {
    if (_amplitude == Rx<Amplitude?>(null)) {
      ampl.value = {'max': intiMax, 'current': initCurrent};
    } else {
      ampl.value = {
        'max': 1.0 - max(0.2, _amplitude.value!.max.abs() / maxAmplitude),
        'current':
            1.0 - max(0.2, _amplitude.value!.current.abs() / maxAmplitude)
      };
    }
  }

  @override
  void onInit() {
    _audioRecorder = AudioRecorder();
    _stateSub = _audioRecorder.onStateChanged().listen((state) {
      _recordState.value = state;
    });

    // Replace with your desired duration
    const duration = Duration(milliseconds: 100);
    setAmplitude(duration);

    // 원하는 녹음 품질과 옵션 설정
    recordConfig = RecordConfig(
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

  Rx<Amplitude?> getAmplitude() {
    return _amplitude;
  }

  Future<String?> stopRecoding() async {
    String? filepath;

    if (await _audioRecorder.isRecording()) {
      filepath = await _audioRecorder.stop();
    }

    return filepath;
  }

  void startRecoding() async {
    if (await _audioRecorder.isRecording()) return;

    final dtByInt = DateTime.timestamp().millisecondsSinceEpoch;
    final temppath = await getTemporaryDirectory();

    final fullpath = '${temppath.path}/$dtByInt.wav';
    print("권한 확인 전 저장 경로는 ${fullpath}입니다");
    if (await _audioRecorder.hasPermission()) {
      print("녹음 중");
      await _audioRecorder.start(
        recordConfig,
        path: fullpath,
      );
    }
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
