import 'package:cloud_firestore/cloud_firestore.dart';

class ScriptModel {
  String? id;
  final String title;
  final String category;
  final List<String> content;
  final Timestamp createdAt;
  
  ScriptModel({
      this.id,
      required this.title,
      required this.category,
      required this.content,
      required this.createdAt,
  });

  ScriptModel.fromDocument({required DocumentSnapshot<Map<String, dynamic>> doc})
      : id = doc.id,
        title = doc.data()!['title'],
        category = doc.data()!['category'],
        content = doc.data()!['content'] == null
          ? []
          : doc.data()!['content'].cast<String>(),
        createdAt = doc.data()!['createdAt'];
        

  Map<String, dynamic> convertToDocument() {
    return {
      'title': title,
      'category': category,
      'content': content,
      'createdAt': createdAt,
    };
  }
}