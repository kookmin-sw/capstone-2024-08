import 'package:capstone/model/google_sign_in_api.dart';
import 'package:capstone/screen/authentication/setup_user.dart';
import 'package:capstone/screen/authentication/social_login.dart';
import 'package:capstone/screen/bottom_navigation.dart';
import 'package:capstone/screen/authentication/controller/user_controller.dart';
import 'package:get/get.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:capstone/constants/text.dart' as text;
import 'package:capstone/constants/color.dart' as colors;

class AuthController extends GetxController {
  static AuthController instance = Get.find();

  late Rx<User?> _user;
  FirebaseAuth authentication = FirebaseAuth.instance;

  @override
  void onReady() {
    super.onReady();
    _user = Rx<User?>(authentication.currentUser);
    _user.bindStream(authentication.userChanges());

    ever(_user, _moveToPage);
  }

  void handleUserInfoCompletion() async {
    AuthController.instance._moveToPage(FirebaseAuth.instance.currentUser);
  }

  _moveToPage(User? user) async {
    if (user == null) {
      Get.offAll(() => const SocialLogin());
    } else {
      bool isUserSetUp = await checkIfUserSetUp(user.uid);

      if (isUserSetUp) {
        Get.put(UserController(), permanent: true);
        await Future.delayed(const Duration(seconds: 2));
        Get.offAll(() => const BottomNavBar());
      } else {
        Get.offAll(() => const SetupUser());
      }
    }
  }

  Future<bool> checkIfUserSetUp(String userId) async {
    try {
      DocumentSnapshot snapshot =
          await FirebaseFirestore.instance.collection('user').doc(userId).get();

      if (snapshot.exists) {
        Map<String, dynamic>? userData =
            snapshot.data() as Map<String, dynamic>?;
        if (userData != null) {
          return true;
        }
      }
      return false;
    } catch (e) {
      if (kDebugMode) {
        print('Error checking user setup: $e');
      }
      return false;
    }
  }

  void logout() {
    authentication.signOut();
  }

  Future<void> deleteUser() async {
    try {
      var googleUser = await GoogleSignInApi.login();
      var user = authentication.currentUser;

      if (user != null && googleUser != null) {
        OAuthCredential? credential;
        var googleAuth = await googleUser.authentication;

        if (user.providerData
            .any((userInfo) => userInfo.providerId == 'google.com')) {
          credential = GoogleAuthProvider.credential(
            accessToken: googleAuth.accessToken,
            idToken: googleAuth.idToken,
          );
        }

        if (credential != null) {
          await user.reauthenticateWithCredential(credential);
        }

        await user.delete();
      }
    } catch (e) {
      _handleError(e);
    }
  }

  Future<UserCredential?> loginWithGoogle(BuildContext context) async {
    try {
      UserCredential? userCredential = await _signInWithCredential(() async {
        var user = await GoogleSignInApi.login();
        var googleAuth = await user!.authentication;
        return GoogleAuthProvider.credential(
          accessToken: googleAuth.accessToken,
          idToken: googleAuth.idToken,
        );
      });

      return userCredential;
    } catch (e) {
      _handleError(e);
      print(e);
    }
    return null;
  }

  Future<UserCredential> _signInWithCredential(
      Future<AuthCredential> Function() getCredential) async {
    var credential = await getCredential();
    return FirebaseAuth.instance.signInWithCredential(credential);
  }

  void _handleError(dynamic e) {
    Get.snackbar(
      text.registrationFailedText,
      e.toString(),
      backgroundColor: colors.errorColor,
      snackPosition: SnackPosition.BOTTOM,
      titleText: const Text(text.registrationFailedText,
          style: TextStyle(color: Colors.white)),
      messageText:
          Text(e.toString(), style: TextStyle(color: Colors.white)),
    );
  }
}
