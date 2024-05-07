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
  );
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
      initialRoute: '/bottom_navigation',
      getPages: [
        GetPage(
          name: '/bottom_navigation', 
          page: () => const BottomNavBar()
        )
      ],
      debugShowCheckedModeBanner: false,
    );
  }
}