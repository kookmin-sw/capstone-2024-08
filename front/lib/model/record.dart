import 'package:cloud_firestore/cloud_firestore.dart';

class RecordModel {
  String? id;
  List<int> scrapSentence;
  List<Map<String, dynamic>>? promptResult;
  Timestamp? practiceDate;
  int? precision;
  
  RecordModel({
      this.id,
      required this.scrapSentence,
      required this.promptResult,
  }); 

  RecordModel.fromDocument({required DocumentSnapshot<Map<String, dynamic>> doc})
      : id = doc.id,
        scrapSentence = doc.data()!['scrapSentence'].cast<int>(),
        promptResult = doc.data()!['promptResult'].cast<Map<String, dynamic>>();
}