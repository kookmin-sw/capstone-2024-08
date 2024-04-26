import 'package:cloud_firestore/cloud_firestore.dart';

class ScriptModel {
  String? id;
  //final String uid;
  final String title;
  final String category;
  final List<String> content;
  final Timestamp createdAt;
  //List<String> keyword;
  
  ScriptModel({
      this.id,
      //required this.uid,
      required this.title,
      required this.category,
      required this.content,
      required this.createdAt,
      //required this.keyword
  });

  ScriptModel.fromDocument({required DocumentSnapshot<Map<String, dynamic>> doc})
      : id = doc.id,
        //uid = doc.data()!['uid'],
        title = doc.data()!['title'],
        category = doc.data()!['category'],
        content = doc.data()!['content'].cast<String>(),
        createdAt = doc.data()!['createdAt'];
        //keyword = doc.data()!['keyword'] == null
        //    ? null
        //    : doc.data()!['keyword'].cast<String>();
        

  Map<String, dynamic> convertToDocument() {
    return {
      //'uid': uid,
      'title': title,
      'category': category,
      'content': content,
      'createdAt': createdAt,
      //'keyword' : keyword
    };
  }
}