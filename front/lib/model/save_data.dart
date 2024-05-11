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
    firestore
        .collection('user_script')
        .doc(user!.uid)
        .collection('script')
        .add(script.convertToDocument());
  }

  Future<Map<String, String>> uploadWavFiles(
      String uid, Map<String, File> wavs) async {
    Map<String, String> urls = {};

    for (MapEntry<String, File> element in wavs.entries) {
      var wavRef = storage.ref().child('user_voice/$uid/${element.key}.wav');
      File file = File(element.value.path);

      try {
        await wavRef.putFile(file);
        String url = await wavRef.getDownloadURL();
        urls[element.key] = url;
        // ignore: empty_catches
      } on FirebaseException {}
    }

    return urls;
  }
}
