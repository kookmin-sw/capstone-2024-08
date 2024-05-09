import 'package:capstone/screen/authentication/controller/auth_controller.dart';
import 'package:capstone/screen/authentication/setup_user.dart';
import 'package:capstone/screen/authentication/social_login.dart';
import 'package:capstone/screen/bottom_navigation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:get/get.dart';
import 'firebase_options.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:capstone/constants/color.dart' as colors;
void main() async {
  await dotenv.load(fileName: '.env');
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  ).then((value) {
    Get.put(AuthController());
  });
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Capstone',
      theme: ThemeData(
        fontFamily: 'KoddiUDOnGothic',
        scaffoldBackgroundColor: colors.bgrBrightColor
      ),
      initialRoute: '/login',
      getPages: [
        GetPage(
          name: '/login', 
          page: () => const SocialLogin()
        ),
        GetPage(
          name: '/user', 
          page: () => const SetupUser()
        ),
        GetPage(
          name: '/bottom_nav', 
          page: () => const BottomNavBar()
        )
      ],
      debugShowCheckedModeBanner: false,
    );
  }
}