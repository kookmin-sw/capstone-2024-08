import 'dart:io';
import 'package:capstone/model/script.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_storage/firebase_storage.dart';

class SaveData {
  FirebaseFirestore firestore = FirebaseFirestore.instance;
  FirebaseStorage storage = FirebaseStorage.instance;
  final User? user = FirebaseAuth.instance.currentUser;

  addUserScript(ScriptModel script) async {
    DocumentReference docRef = await firestore
        .collection('user_script')
        .doc(user!.uid)
        .collection('script')
        .add(script.convertToDocument());

    // Get the document ID
    String documentId = docRef.id;
    script.id = documentId;
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

  Future<void> updateOneSentencePracticeResult(
      {required String scriptId,
      required String scriptType,
      List<int>? scrapSentence}) async {
    User? user = FirebaseAuth.instance.currentUser;

    if (user != null) {
      await FirebaseFirestore.instance
          .collection('user')
          .doc(user.uid)
          .collection('${scriptType}_practice')
          .doc(scriptId)
          .set({'scrapSentence': scrapSentence});
    }
  }

  Future<void> updatePromptPracticeResult(
      {required String scriptId,
      required String scriptType,
      int? precision}) async {
    User? user = FirebaseAuth.instance.currentUser;
    Map<String, dynamic> promptResult = {
      'practiceDate': Timestamp.now(),
      'precision': precision
    };

    if (user != null) {
      DocumentReference docRef = FirebaseFirestore.instance
          .collection('user')
          .doc(user.uid)
          .collection('${scriptType}_practice')
          .doc(scriptId);

      // Fetch the document
      DocumentSnapshot docSnapshot = await docRef.get();

      if (docSnapshot.exists) {
        // If document exists, update the field
        List<Map<String, dynamic>> existingPromptResult =
            List<Map<String, dynamic>>.from(
                docSnapshot.get('promptResult') ?? []);
        existingPromptResult.add(promptResult);
        await docRef.update(
            {'promptResult': FieldValue.arrayUnion(existingPromptResult)});
      } else {
        // If document does not exist, create the document with the field
        await docRef.set({'promptResult': promptResult});
      }
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
      } on FirebaseException catch (e) {
        print("========================");
        print("Failed with error '${e.code}': ${e.message}");
      }
    }

    return urls;
  }

  updateAttendance(String uid, int attendanceStreak) async {
    await firestore.collection('user').doc(uid).update({
      'lastAccessDate': Timestamp.now(),
      'attendanceStreak': attendanceStreak
    });
  }

  Future<DocumentReference> updateLastPracticeScript(
      String uid, String scriptType, String scriptId) async {
    DocumentReference scriptRef = FirebaseFirestore.instance
        .collection('${scriptType}_script')
        .doc(scriptId);
    await firestore
        .collection('user')
        .doc(uid)
        .update({'lastPracticeScript': scriptRef});

    return scriptRef;
  }

  Future<List<int>?> scrap(
      String scriptType, String scriptId, String uid, int sentenceIndex) async {
    try {
      DocumentReference scriptRef = FirebaseFirestore.instance
          .collection('user')
          .doc(uid)
          .collection('${scriptType}_practice')
          .doc(scriptId);

      await FirebaseFirestore.instance.runTransaction((transaction) async {
        DocumentSnapshot scriptDoc = await transaction.get(scriptRef);

        if (scriptDoc.exists) {
          List<int>? scrapSentence =
              List.from(scriptDoc.get('scrapSentence') ?? []);
          scrapSentence.add(sentenceIndex);
          transaction.update(scriptRef, {'scrapSentence': scrapSentence});

          return scrapSentence;
        }
      });
    } catch (e) {
      print('Error adding value to scrap sentence: $e');
    }
    return null;
  }

  Future<List<int>?> cancelScrap(
      String scriptType, String scriptId, String uid, int sentenceIndex) async {
    try {
      DocumentReference scriptRef = FirebaseFirestore.instance
          .collection('user')
          .doc(uid)
          .collection('${scriptType}_practice')
          .doc(scriptId);

      await FirebaseFirestore.instance.runTransaction((transaction) async {
        DocumentSnapshot scriptDoc = await transaction.get(scriptRef);

        if (scriptDoc.exists) {
          List<int>? scrapSentence =
              List.from(scriptDoc.get('scrapSentence') ?? []);
          scrapSentence.remove(sentenceIndex);
          transaction.update(scriptRef, {'scrapSentence': scrapSentence});

          return scrapSentence;
        }
      });
    } catch (e) {
      print('Error adding value to scrap sentence: $e');
    }
    return null;
  }
}
