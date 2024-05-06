import 'package:capstone/widget/audio_recoder/recoder_controller.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

void useEffectOnce(void Function() callback) {
  final once = true.obs; // RxBool로 한 번만 실행될 때 true로 설정

  // once 값이 변할 때마다 callback을 실행하도록 설정
  ever(once, (_) {
    if (once.value) {
      callback();
      once.value = false; // 한 번 실행 후에는 false로 변경하여 다시 실행되지 않도록 함
    }
  });
}

class RecordingSheet extends StatefulWidget {
  const RecordingSheet({super.key});

  @override
  State<RecordingSheet> createState() => _RecordingSheetState();
}

class _RecordingSheetState extends State<RecordingSheet> {
  final RecorderController recorderController = Get.put(RecorderController());
  static const double sheetHeight = 250.0;

  @override
  Widget build(BuildContext context) {
    /// BottomSheet가 열리면 바로 음성녹음 시작
    useEffectOnce(() => recorderController.startRecoding());

    /// duration에 따라 amplitude 설정
    recorderController.setAmplitude(const Duration(milliseconds: 100));

    ///  추가적으로 초기값은 20%(0.2) 정도로 적당히 설정합니다.
    recorderController.updateAmpl(intiMax: 0.2, initCurrent: 0.2);

    // print('max: ${ampl.value.max}, cur: ${ampl.value.cur}');

    return Container(
      alignment: Alignment.center,
      height: sheetHeight,
      decoration: const BoxDecoration(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(16.0),
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        fit: StackFit.expand,
        alignment: Alignment.center,
        children: [
          Column(
            children: [
              Expanded(
                flex: 2,
                child: AnimatedContainer(
                  duration: Duration(milliseconds: 200),
                  curve: Curves.easeIn,
                  alignment: Alignment.center,
                  child: Container(
                    height: recorderController.ampl.value['current']! *
                        (sheetHeight * 0.5),
                    decoration: BoxDecoration(
                      color: Colors.red.shade300,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              ),
              Expanded(
                flex: 1,
                child: IconButton(
                  icon: const Icon(Icons.stop_circle_outlined, size: 64.0),
                  onPressed: () async {
                    final filepath = await recorderController.stopRecoding();
                    debugPrint("--------------");

                    debugPrint(filepath);

                    if (filepath != null) {
                      Navigator.pop(context, filepath);
                    } else {
                      Future(
                        () => ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Recording Failed'),
                          ),
                        ),
                      ).then((value) => Navigator.pop(context));
                    }
                  },
                ),
              ),
              SizedBox(height: MediaQuery.of(context).padding.bottom),
            ],
          ),
        ],
      ),
    );
  }
}
