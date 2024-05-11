import 'dart:io';
import 'package:capstone/model/script.dart';
import 'package:capstone/model/user.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';

class SaveData {
  FirebaseFirestore firestore = FirebaseFirestore.instance;
  FirebaseStorage storage = FirebaseStorage.instance;

  addUserScript(String uid, ScriptModel script) async {
    firestore
        .collection('user_script')
        .doc(uid)
        .collection('script')
        .add(script.convertToDocument());
  }

  Future<void> saveUserInfo({
    required String nickname,
    required String character,
    Timestamp? lastAccessDate,
    int? attendanceStreak,
    Map<String, String>? voiceUrls,
    DocumentReference? lastPracticeScript,
  }) async {
    User? user = FirebaseAuth.instance.currentUser;

    if (user != null) {
      await FirebaseFirestore.instance.collection('user').doc(user.uid).set({
        'nickname': nickname,
        'character': character,
        'attendanceStreak': attendanceStreak,
        'lastAccessDate': lastAccessDate,
        'lastPracticeScript': lastPracticeScript,
        'voiceUrls': voiceUrls
      });
    }
  }

  Future<Map<String, String>> uploadWavFiles(
      String uid, Map<String, String> wavs) async {
    Map<String, String> urls = {};

    for (MapEntry<String, String> element in wavs.entries) {
      var wavRef = storage.ref().child('user_voice/$uid/${element.key}.wav');
      File file = File(element.value);

      try {
        await wavRef.putFile(file);
        String url = await wavRef.getDownloadURL();
        urls[element.key] = url;
        // ignore: empty_catches
      } on FirebaseException {}
    }

    return urls;
  }

  updateAttendance(String uid, int attendanceStreak) async {
    await firestore.collection('user').doc(uid).set({
      'lastAccessDate': Timestamp.now(),
      'attendanceStreak': attendanceStreak
    });
  }
}
