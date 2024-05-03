import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:capstone/screen/bottom_navigation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:get/get.dart';
import 'package:responsive_framework/responsive_framework.dart';
import 'firebase_options.dart';
import 'package:firebase_core/firebase_core.dart';

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
      theme: ThemeData(fontFamily: fonts.font),
      builder: (context, child) =>
          ResponsiveBreakpoints.builder(child: child!, breakpoints: [
        const Breakpoint(start: 0, end: 450, name: MOBILE),
        const Breakpoint(start: 451, end: 800, name: TABLET),
      ]),
      initialRoute: '/bottom_navigation',
      getPages: [
        GetPage(name: '/bottom_navigation', page: () => const BottomNavBar())
      ],
      debugShowCheckedModeBanner: false,
    );
  }
}
