import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

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
    return firestore
      .collection('example_script')
      .snapshots()
      .map((snapshot) => snapshot.docs
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

  Future<DocumentSnapshot<Map<String, dynamic>>> readRecordDocument(String scriptType, String documentId) async {
    return await firestore.collection('user').doc('mg').collection('${scriptType}_practice').doc(documentId).get();
  }
}
