import 'package:capstone/widget/basic_app_bar.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

class Policy extends StatelessWidget {
  const Policy({
    Key? key,
    required this.policy
  }) : super(key: key);

  final Map<String, String> policy;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: basicAppBar(title: policy['policyName']!),
      body: SingleChildScrollView(
        child: Container(
          padding: const EdgeInsets.all(10),
          child: Wrap(
            children: [
                Text(
                  policy['policyContent']!,
                  semanticsLabel: policy['policyContent'],
                  style: const TextStyle(
                    fontSize: 15,
                    color: colors.textColor,
                    fontWeight: FontWeight.w400,
                  )
                )
          ])
      )),
    );
  }
}
