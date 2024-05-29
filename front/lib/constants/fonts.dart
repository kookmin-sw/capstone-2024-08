import 'package:capstone/widget/utils/device_size.dart';
import 'package:flutter/material.dart';

const font = 'KoddiUDOnGothic';

//font size
//const double plainText(context) = 15;
//const double title = 17;
//const double category = 16;
const double tab = 20;
const double button = 16;

double plainText(BuildContext context) {
  return getDeviceWidth(context) * 0.038;
}

double title(BuildContext context) {
  return getDeviceWidth(context) * 0.04;
}

double category(BuildContext context) {
  return getDeviceWidth(context) * 0.039;
}