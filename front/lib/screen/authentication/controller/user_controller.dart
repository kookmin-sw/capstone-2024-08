import 'dart:io';

import 'package:capstone/model/load_data.dart';
import 'package:capstone/model/save_data.dart';
import 'package:capstone/model/user.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';

class UserController extends GetxController {
  final User? user = FirebaseAuth.instance.currentUser;
  final LoadData loadData = LoadData();
  final SaveData saveData = SaveData();

  final RxBool userModelReady = false.obs;

  late UserModel userModel;
  late Map<String, File?> wavFiles;

  @override
  void onInit() async {
    super.onInit();
    DocumentSnapshot<Map<String, dynamic>> document =
        await loadData.readUser(uid: user!.uid);
    userModel = UserModel.fromDocument(doc: document);
    userModelReady.value = true;
    await updateAttendance();
    wavFiles = {};
    await downloadAllWavFiles();
  }

  Future<void> updateAttendance() async {
    if (userModel.lastAccessDate != null) {
      DateTime lastAccessDate = userModel.lastAccessDate!.toDate();
      DateTime currentDate = DateTime.now();
      if (_isSameDay(lastAccessDate, currentDate)) {
        // 같은 날 일 경우에는 업데이트 안해줘도 되지만 다운로드 음성 파일의 디버깅을 위해 임시로 추가해둠
        await saveData.updateAttendance(user!.uid, userModel.attendanceStreak!);
        return;
      } else if (_isConsecutiveDay(lastAccessDate, currentDate)) {
        userModel.attendanceStreak = userModel.attendanceStreak! + 1;
      } else {
        userModel.attendanceStreak = 1;
      }
    } else {
      userModel.attendanceStreak = 1;
    }
    saveData.updateAttendance(user!.uid, userModel.attendanceStreak!);
  }

  bool _isConsecutiveDay(DateTime lastDate, DateTime currentDate) {
    return (currentDate.year == lastDate.year &&
        currentDate.month == lastDate.month &&
        currentDate.day - lastDate.day == 1);
  }

  bool _isSameDay(DateTime lastDate, DateTime currentDate) {
    return (currentDate.year == lastDate.year &&
        currentDate.month == lastDate.month &&
        currentDate.day == lastDate.day);
  }

  Future<void> downloadAllWavFiles() async {
    if (userModel.voiceUrls != null) {
      for (MapEntry<String, String> element in userModel.voiceUrls!.entries) {
        wavFiles[element.key] =
            await loadData.downloadWavFile(element.value, element.key);
      }
    }
  }
}
