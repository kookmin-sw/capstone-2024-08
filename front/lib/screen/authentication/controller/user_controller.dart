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
    await downloadAllWavFiles();
  }

  void updateAttendance() {
    if (userModel.lastAccessDate != null) {
      DateTime lastAccessDate = userModel.lastAccessDate!.toDate();
      DateTime currentDate = DateTime.now();
      if (_isConsecutiveDay(lastAccessDate, currentDate)) {
        if (userModel.attendanceStreak == null) {
          userModel.attendanceStreak = 1;
        } else {
          userModel.attendanceStreak = userModel.attendanceStreak! + 1;
        }
        saveData.updateAttendance(user!.uid, userModel.attendanceStreak!);
      }
    }
  }

  bool _isConsecutiveDay(DateTime lastDate, DateTime currentDate) {
    return (currentDate.year == lastDate.year &&
        currentDate.month == lastDate.month &&
        currentDate.day - lastDate.day == 1);
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
