import 'package:capstone/screen/authentication/controller/auth_controller.dart';
import 'package:capstone/widget/basic_app_bar.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:capstone/widget/warning_dialog.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/route.dart' as routes;

class Setting extends StatelessWidget {
  const Setting({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);
    var deviceHeight = getDeviceHeight(context);

    return Scaffold(
      appBar: basicAppBar(title: '설정'),
      body: Container(
        padding: EdgeInsets.all(deviceWidth * 0.05),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: routes.settingItems.map((item) =>
            GestureDetector(
              onTap: () {
                if (item.containsKey('action')) {
                  item['action'](context);
                }
                if (item.containsKey('route')) {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => item['route'],
                    ),
                  );
                }
              },
              child: Container(
                width: deviceWidth,
                height: deviceHeight * 0.1,
                margin: EdgeInsets.only(bottom: deviceWidth * 0.04),
                decoration: BoxDecoration(
                  color: colors.blockColor,
                  borderRadius: BorderRadius.circular(15),
                  boxShadow: const [
                    BoxShadow(
                      color: colors.buttonSideColor,
                      blurRadius: 2,
                      spreadRadius: 2,
                    )
                  ]),
                padding: const EdgeInsets.all(15),
                child: Center(
                  child: Text(
                    item['name'],
                    semanticsLabel: item['name'],
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w800,
                      color: colors.textColor
                    ),
                  )
                ),
              ))
          ).toList()
        )
    ));
  }
}

Future<bool> showConfirmationDialog(BuildContext context, String type) async {
    return await showDialog(
      context: context,
      builder: (BuildContext context) =>
        WarningDialog(
          warningObject: type,
        )
    );
  }


  Future<void> handleLogoutAction(BuildContext context) async {
    AuthController authController = AuthController.instance;
    bool confirmLogout = await showConfirmationDialog(context, 'logout');

    if (confirmLogout) {
      authController.logout();
    }
  }

  void handleDeleteAction(BuildContext context) async {
    AuthController authController = AuthController.instance;
    bool confirmDelete = await showConfirmationDialog(context, 'deleteUser');

    if (confirmDelete) {
      CollectionReference userInfo = FirebaseFirestore.instance.collection('user');
      User? user = FirebaseAuth.instance.currentUser;
      
      if (user != null) {
        userInfo.doc(user.uid).delete();
        authController.deleteUser(); 
      }
    }
  }