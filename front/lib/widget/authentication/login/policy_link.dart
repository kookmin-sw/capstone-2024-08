import 'package:capstone/screen/setting/policy.dart';
import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:get/get.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

class PolicyLink extends StatelessWidget {
  final String text;
  final Map<String, String> policyPath;
  final SvgPicture icon;

  const PolicyLink({
    Key? key,
    required this.text,
    required this.policyPath,
    required this.icon,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    var deviceHeight = getDeviceHeight(context);

    return TextButton(
      onPressed: () {
        Get.to(() => Policy(policy: policyPath));
      },
      style: TextButton.styleFrom(
        padding: EdgeInsets.zero,
      ),
      child: Column(
        children: [
          Container(
            margin: EdgeInsets.only(bottom: deviceHeight * 0.004),
            child: Text(
              text,
              style: TextStyle(
                fontSize: deviceHeight * 0.018,
                color: colors.blockColor,
                fontWeight: FontWeight.w400,
              ),
            ),
          ),
          icon,
        ],
      ),
    );
  }
}
