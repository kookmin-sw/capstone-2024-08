import 'dart:io';

import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

class LoadData {
  FirebaseFirestore firestore = FirebaseFirestore.instance;

  Future<DocumentSnapshot<Map<String, dynamic>>> readUser(
      {required String uid}) async {
    var userDocumentSnapshot =
        await firestore.collection('user').doc(uid).get();
    print("-----------");
    print(userDocumentSnapshot);
    return userDocumentSnapshot;
  }

  Future<ScriptModel?> readScriptByDocumentRef(
      DocumentReference documentRef) async {
    try {
      DocumentSnapshot<Map<String, dynamic>> snapshot =
          await documentRef.get() as DocumentSnapshot<Map<String, dynamic>>;

      if (snapshot.exists) {
        return ScriptModel.fromDocument(doc: snapshot);
      } else {
        return null;
      }
    } catch (e) {
      print('Error fetching script: $e');
      return null;
    }
  }

  Stream<List<ScriptModel>> readExampleScripts(String? category) {
    if (category == '전체') {
      return firestore
          .collection('example_script')
          .orderBy('createdAt', descending: true)
          .snapshots()
          .map((snapshot) => snapshot.docs
              .map((doc) => ScriptModel.fromDocument(doc: doc))
              .toList());
    } else {
      return firestore
          .collection('example_script')
          .where('category', isEqualTo: category)
          .orderBy('createdAt', descending: true)
          .snapshots()
          .map((snapshot) => snapshot.docs
              .map((doc) => ScriptModel.fromDocument(doc: doc))
              .toList());
    }
  }

  Stream<List<ScriptModel>> readUserScripts(String? category) {
    if (category == '전체') {
      return firestore
          .collection('user_script')
          .doc('mg')
          .collection('script')
          .orderBy('createdAt', descending: true)
          .snapshots()
          .map((snapshot) => snapshot.docs
              .map((doc) => ScriptModel.fromDocument(doc: doc))
              .toList());
    } else {
      return firestore
          .collection('user_script')
          .doc('mg')
          .collection('script')
          .where('category', isEqualTo: category)
          .orderBy('createdAt', descending: true)
          .snapshots()
          .map((snapshot) => snapshot.docs
              .map((doc) => ScriptModel.fromDocument(doc: doc))
              .toList());
    }
  }

  Stream<List<ScriptModel>> searchExampleScript(String? query) {
    return firestore.collection('example_script').snapshots().map((snapshot) =>
        snapshot.docs
            .where((doc) => doc['title'].toString().contains(query ?? ''))
            .map((doc) => ScriptModel.fromDocument(doc: doc))
            .toList());
  }

  Stream<List<ScriptModel>> searchUserScript(String? query) {
    return firestore
        .collection('user_script')
        .doc('mg')
        .collection('script')
        .snapshots()
        .map((snapshot) => snapshot.docs
            .where((doc) => doc['title'].toString().contains(query ?? ''))
            .map((doc) => ScriptModel.fromDocument(doc: doc))
            .toList());
  }

  Stream<List<RecordModel>> readUserPracticeRecord(String scriptType) {
    return firestore
        .collection('user')
        .doc('mg')
        .collection('${scriptType}_practice')
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => RecordModel.fromDocument(doc: doc))
            .toList());
  }

  Future<DocumentSnapshot<Map<String, dynamic>>> readRecordDocument(
      String scriptType, String documentId) async {
    return await firestore
        .collection('user')
        .doc('mg')
        .collection('${scriptType}_practice')
        .doc(documentId)
        .get();
  }

  Future<File?> downloadWavFile(String filePath, String fileNanme) async {
    try {
      // 파일 경로를 기반으로 Firebase Storage에서 파일을 다운로드
      Reference ref = FirebaseStorage.instance.ref().child(filePath);
      Uint8List? data = await ref.getData();
      Directory dir = await getTemporaryDirectory();
      String localPath = '$dir/user_voices/$fileNanme.wav';

      if (data != null) {
        // 다운로드한 데이터를 사용하여 파일을 생성하거나 저장할 수 있음
        // 여기서는 예시로 로컬 파일에 저장하도록 함
        File file = File(localPath);
        await file.writeAsBytes(data);
        print('File downloaded successfully');
        return file;
      } else {
        print('Failed to download file: Data is null');
      }
    } catch (e) {
      print('Error downloading file: $e');
    }
    return null;
  }
}
