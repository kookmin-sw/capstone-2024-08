import 'package:capstone/model/script.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class LoadData {
  FirebaseFirestore firestore = FirebaseFirestore.instance;
  
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
}
