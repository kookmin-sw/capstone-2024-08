import 'package:capstone/widget/authentication/login/policy_link.dart';
import 'package:capstone/widget/authentication/login/social_login_button.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/text.dart' as text;
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/auth_icons.dart' as auth_icons;
import 'package:capstone/screen/authentication/controller/auth_controller.dart';

class SocialLogin extends StatelessWidget {
  const SocialLogin({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    var deviceWidth = MediaQuery.of(context).size.width;
    var deviceHeight = MediaQuery.of(context).size.height;

    Widget buildLogo() {
      return Container(
        width: deviceWidth * 0.5,
        height: deviceHeight * 0.38,
        decoration: const BoxDecoration(
          color: Colors.transparent,
        ),
        child: auth_icons.loroLogo,
      );
    }

    Widget buildButtons(BuildContext context) {
      return SocialLoginButton(
        color: Colors.white,
        icon: auth_icons.googleIcon,
        text: text.googleLoginText,
        onPressed: (context) async {
          await AuthController.instance.loginWithGoogle(context);
        },
      );
    }

    Widget buildPolicyLinks() {
      return Column(
        children: [
          PolicyLink(
            text: text.termsOfService,
            policyPath: text.usingPolicy,
            icon: auth_icons.tosLine,
          ),
          SizedBox(height: deviceHeight * 0.01),
          PolicyLink(
            text: text.privacyPolicy,
            policyPath: text.personalData,
            icon: auth_icons.policyLine,
          ),
        ],
      );
    }


    Widget buildScaffoldBody(BuildContext context) {
      return Container(
        color: colors.bgrDarkColor,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            buildLogo(),
            SizedBox(height: deviceHeight * 0.08),
            buildButtons(context),
            SizedBox(height: deviceHeight * 0.05),
            buildPolicyLinks()
          ],
        ),
      );
    }

    return buildScaffoldBody(context);
  }
}
