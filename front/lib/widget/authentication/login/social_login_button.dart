import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

class SocialLoginButton extends StatelessWidget {
  final Color color;
  final Widget icon;
  final String text;
  final void Function(BuildContext) onPressed;

  const SocialLoginButton({
    Key? key,
    required this.color,
    required this.icon,
    required this.text,
    required this.onPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    var deviceWidth = getDeviceWidth(context);
    var deviceHeight = getDeviceHeight(context);

    return Container(
      width: deviceWidth * 0.91,
      height: deviceHeight * 0.06,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(deviceWidth * 0.11),
        color: color,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 6,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: TextButton(
        onPressed: () async {
          onPressed(context);
        },
        child: Center(
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: EdgeInsets.only(right: deviceWidth * 0.024),
                child: Container(
                  width: deviceWidth * 0.12,
                  height: deviceHeight * 0.05,
                  decoration: const BoxDecoration(
                    color: Colors.transparent,
                  ),
                  child: icon,
                ),
              ),
              Text(
                text,
                style: TextStyle(
                fontSize: deviceWidth * 0.048,
                color: colors.textColor,
                fontWeight: FontWeight.w400,
              ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
