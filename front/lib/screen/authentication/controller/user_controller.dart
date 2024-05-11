import 'package:capstone/model/load_data.dart';
import 'package:capstone/model/save_data.dart';
import 'package:capstone/model/user.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';

class UserController extends GetxController {
  final User? user = FirebaseAuth.instance.currentUser;

  final LoadData loadData = LoadData();
  final SaveData saveData = SaveData();

  final RxBool userModelReady = false.obs;

  late UserModel userModel;

  @override
  void onInit() async {
    super.onInit();
    DocumentSnapshot<Map<String, dynamic>> document =
        await loadData.readUser(uid: 'anzxwon'); // user!.uid
    userModel = UserModel.fromDocument(doc: document);
    userModelReady.value = true;
  }
}
