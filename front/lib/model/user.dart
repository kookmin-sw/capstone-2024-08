import 'package:cloud_firestore/cloud_firestore.dart';

class UserModel {
  String? id;
  String? nickname;
  String? character;
  Timestamp? lastAccessDate;
  int? attendanceStreak;
  Map<String, String>? voiceUrls;
  DocumentReference? lastPracticeScript;

  UserModel(
      {this.id,
      required this.nickname,
      required this.character,
      required this.lastAccessDate,
      required this.attendanceStreak,
      required this.voiceUrls,
      required this.lastPracticeScript});

  // Deserialize from Firestore document snapshot
  UserModel.fromDocument({required DocumentSnapshot<Map<String, dynamic>> doc})
      : id = doc.id,
        nickname = doc.get('nickname'),
        character = doc.get('character'),
        lastAccessDate = doc.get('lastAccessDate'),
        attendanceStreak = doc.get('attendanceStreak'),
        voiceUrls = doc.data()!['voiceUrls'] == null
            ? null
            : (doc.data()!['voiceUrls'] as Map<String, dynamic>)
                .cast<String, String>(),
        lastPracticeScript = doc.get('lastPracticeScript');

  // Serialize to Firestore document data
  Map<String, dynamic> convertToDocument() {
    return {
      'nickname': nickname,
      'character': character,
      'lastAccessDate': lastAccessDate,
      'attendanceStreak': attendanceStreak,
      'voiceUrls': voiceUrls,
      'lastPracticeScript': lastPracticeScript
    };
  }

  // Helper function to cast Map<String, dynamic> to Map<String, String>
  Map<String, String>? castMapOfStringString(Map<String, dynamic>? input) {
    if (input == null) {
      return null;
    }

    Map<String, String> result = {};
    input.forEach((key, value) {
      if (value is String) {
        result[key] = value;
      }
    });
    return result.isNotEmpty ? result : null;
  }
}
