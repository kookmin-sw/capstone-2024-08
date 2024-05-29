import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/fonts.dart' as fonts;

class PromptPrecisionGraph extends StatefulWidget {
  const PromptPrecisionGraph({
    super.key,
    required this.promptResult
  });

  final List<Map<String, dynamic>> promptResult;

  @override
  State<PromptPrecisionGraph> createState() => _PromptPrecisionGraphState();
}

class _PromptPrecisionGraphState extends State<PromptPrecisionGraph> {
  List<Color> gradientColors = [
    colors.precisionGraphBgrColor,
    colors.precisionGraphLineColor,
  ];

  double timestampToDouble(Timestamp timestamp) {
    return timestamp.toDate().millisecondsSinceEpoch.toDouble();
  }

  String datePadLeft(int date){
    return date.toString().padLeft(2, '0');
  }

  String getPracticeDate(double value) {
    DateTime date = DateTime.fromMillisecondsSinceEpoch(value.toInt());
    return '${date.month}/${date.day} ${datePadLeft(date.hour)}:${datePadLeft(date.minute)}\n';
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.fromLTRB(10, 15, 15, 15),
      decoration: BoxDecoration(
        color: colors.blockColor,
        borderRadius: BorderRadius.circular(10),
        boxShadow: const [
          BoxShadow(
            color: colors.buttonSideColor,
            blurRadius: 5,
            spreadRadius: 3,
          )
        ],
      ),
      child: AspectRatio(
        aspectRatio: 1.60,
        child: LineChart(
            graphData(),
        ),
      ),
    );
  }

  Widget leftTitleWidget(double value, TitleMeta meta) {
    TextStyle style = TextStyle(
      fontWeight: FontWeight.w500,
      fontSize: fonts.plainText(context),
      color: colors.textColor
    );
    String precision;
    switch (value.toInt()) {
      case 0:
        precision = '0';
      case 25:
        precision = '25';
        break;
      case 50:
        precision = '50';
        break;
      case 75:
        precision = '75';
        break;
      case 100:
        return SideTitleWidget(
          axisSide: meta.axisSide,
          child: Text('100', 
          style: TextStyle(
            fontWeight: FontWeight.w500,
            fontSize: fonts.plainText(context) * 0.9,
            color: colors.textColor
          ), 
          textAlign: TextAlign.center),
        );
      default:
        return Container();
    }

    return SideTitleWidget(
      axisSide: meta.axisSide,
      child: Text(precision, style: style, textAlign: TextAlign.center),
    );
  }

  LineChartData graphData() {
    return LineChartData(
      gridData: const FlGridData(show: false),
      borderData: FlBorderData(show: false),
      titlesData: FlTitlesData(
        show: true,
        rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        bottomTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        leftTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            interval: 25,
            reservedSize: 35,
            getTitlesWidget: leftTitleWidget,
          ),
        ),
      ),
      minX: timestampToDouble(widget.promptResult.first['practiceDate']),
      maxX: timestampToDouble(widget.promptResult.last['practiceDate']),
      minY: 0,
      maxY: 100,
      lineBarsData: [
        LineChartBarData(
          spots: [
            for (Map<String, dynamic> result in widget.promptResult)
              FlSpot(timestampToDouble(result['practiceDate']), result['precision'].toDouble()),
          ],
          isCurved: true,
          gradient: LinearGradient(
            colors: gradientColors,
          ),
          barWidth: 5,
          isStrokeCapRound: true,
          belowBarData: BarAreaData(
            show: true,
            gradient: LinearGradient(
              colors: gradientColors
                  .map((color) => color.withOpacity(0.25))
                  .toList(),
            ),
          ),
        ),  
      ],
      lineTouchData: LineTouchData(
        touchTooltipData: LineTouchTooltipData(
          getTooltipItems: (List<LineBarSpot> touchedBarSpots) {
            return touchedBarSpots.map((flSpot) {
              return LineTooltipItem(
                getPracticeDate(flSpot.x),
                TextStyle(
                  color: colors.precisionGraphBgrColor,
                  fontWeight: FontWeight.w500,
                ),
                children: [
                  TextSpan(
                    text: flSpot.y.toInt().toString(),
                    style: TextStyle(
                      color: colors.precisionGraphBgrColor,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ],
              );
            }).toList();
          },
        ),
      )
    );
  }
}