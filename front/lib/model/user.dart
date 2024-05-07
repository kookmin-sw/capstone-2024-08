import 'package:cloud_firestore/cloud_firestore.dart';

class UserModel {
  String? nickname;
  String? character;
  Timestamp? lastAccessDate;
  int? attendanceStreak;
  Map<String, String>? voiceUrls;
  DocumentReference? lastPracticeScript;

  UserModel(
      {required this.nickname,
      required this.character,
      required this.lastAccessDate,
      required this.attendanceStreak,
      required this.voiceUrls,
      required this.lastPracticeScript});

  // Deserialize from Firestore document snapshot
  UserModel.fromDocument(DocumentSnapshot<Map<String, dynamic>> doc)
      : nickname = doc.get('nickname'),
        character = doc.get('character'),
        lastAccessDate = doc.get('lastAccessDate'),
        attendanceStreak = doc.get('attendanceStreak'),
        voiceUrls = doc.get('voice'),
        lastPracticeScript = doc.get('lastPracticeScript');

  // Serialize to Firestore document data
  Map<String, dynamic> convertToDocument() {
    return {
      'nickname': nickname,
      'character': character,
      'lastAccessDate': lastAccessDate,
      'attendanceStreak': attendanceStreak,
      'voice': voiceUrls,
      'lastPracticeScript': lastPracticeScript
    };
  }
}
