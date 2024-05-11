import 'package:capstone/screen/authentication/controller/auth_controller.dart';
import 'package:capstone/screen/authentication/setup_user.dart';
import 'package:capstone/screen/authentication/social_login.dart';
import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:capstone/screen/bottom_navigation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:get/get.dart';
import 'firebase_options.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'dart:io';
import 'package:capstone/constants/color.dart' as colors;

void main() async {
  await dotenv.load(fileName: '.env');
  getPermission();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  ).then((value) async {
    Get.put(AuthController());
  });
  runApp(const MyApp());
}

Future<bool> getPermission() async {
  Map<Permission, PermissionStatus> permissions;
  if (Platform.isIOS) {
    permissions = await [
      Permission.accessNotificationPolicy,
      Permission.microphone,
      Permission.speech,
      Permission.storage
    ].request();
  } else {
    permissions = await [
      Permission.notification,
      Permission.microphone,
      Permission.speech,
      Permission.storage
    ].request();
  }

  if (permissions.values.every((element) => element.isGranted)) {
    return Future.value(true);
  } else {
    return Future.value(false);
  }
}

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
  print('백그라운드 메시지 처리.. ${message.notification!.body!}');
}

void initializeNotification() async {
  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

  AndroidNotificationChannel channel = const AndroidNotificationChannel(
    'high_importance_channel', // id
    'high_importance_notification', // title
    description:
        'This channel is used for important notifications.', // description
    importance: Importance.max,
  );

  FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  await flutterLocalNotificationsPlugin
      .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin>()
      ?.createNotificationChannel(channel);

  AndroidInitializationSettings androidInitializationSettings =
      const AndroidInitializationSettings('@mipmap/android_app_logo');

  DarwinInitializationSettings iosInitializationSettings =
      const DarwinInitializationSettings(
    requestAlertPermission: false,
    requestBadgePermission: false,
    requestSoundPermission: false,
  );

  InitializationSettings initializationSettings = InitializationSettings(
    android: androidInitializationSettings,
    iOS: iosInitializationSettings,
  );

  await flutterLocalNotificationsPlugin.initialize(initializationSettings);

  await FirebaseMessaging.instance.setForegroundNotificationPresentationOptions(
    alert: true,
    badge: true,
    sound: true,
  );

  await FirebaseMessaging.instance.requestPermission(
    alert: true,
    announcement: false,
    badge: true,
    carPlay: false,
    criticalAlert: false,
    provisional: false,
    sound: true,
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Capstone',
      theme: ThemeData(
          fontFamily: fonts.font,
          scaffoldBackgroundColor: colors.bgrBrightColor),
      initialRoute: '/login',
      getPages: [
        GetPage(name: '/login', page: () => const SocialLogin()),
        GetPage(name: '/user', page: () => const SetupUser()),
        GetPage(name: '/bottom_nav', page: () => const BottomNavBar())
      ],
      debugShowCheckedModeBanner: false,
    );
  }
}
