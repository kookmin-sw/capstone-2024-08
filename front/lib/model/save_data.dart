import 'package:capstone/model/script.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class SaveData {
  FirebaseFirestore firestore = FirebaseFirestore.instance;
  
  addUserScript(String uid, ScriptModel script) async {
    firestore
    .collection('user_script')
    .doc(uid)
    .collection('script')
    .add(script.convertToDocument());
  }
}
