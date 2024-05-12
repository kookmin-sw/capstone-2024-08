import 'package:capstone/widget/authentication/login/policy_link.dart';
import 'package:capstone/widget/authentication/login/social_login_button.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/text.dart' as text;
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/image.dart' as images;
import 'package:capstone/screen/authentication/controller/auth_controller.dart';

class SocialLogin extends StatelessWidget {
  const SocialLogin({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Widget buildLogo() {
      return Container(
        width: getDeviceWidth(context) * 0.5,
        height: getDeviceHeight(context) * 0.38,
        decoration: const BoxDecoration(
          color: Colors.transparent,
        ),
        child: images.loroLogo,
      );
    }

    Widget buildButtons(BuildContext context) {
      return SocialLoginButton(
        color: Colors.white,
        icon: images.googleIcon,
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
            icon: images.tosLine,
          ),
          SizedBox(height: getDeviceHeight(context) * 0.01),
          PolicyLink(
            text: text.privacyPolicy,
            policyPath: text.personalData,
            icon: images.policyLine,
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
            SizedBox(height: getDeviceHeight(context) * 0.08),
            buildButtons(context),
            SizedBox(height: getDeviceHeight(context) * 0.05),
            buildPolicyLinks()
          ],
        ),
      );
    }

    return buildScaffoldBody(context);
  }
}
