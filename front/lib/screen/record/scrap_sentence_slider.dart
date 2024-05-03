import 'package:capstone/widget/script/script_content_block.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:flutter/material.dart';
import 'package:dots_indicator/dots_indicator.dart';
import 'package:capstone/constants/color.dart' as colors;

class ScrapSentenceSlider extends StatefulWidget {
  const ScrapSentenceSlider({
    super.key, 
    required this.scrapSentenceList
  });

  final List<String> scrapSentenceList;

  @override
  State<ScrapSentenceSlider> createState() => _ScrapSentenceSliderState();
}

class _ScrapSentenceSliderState extends State<ScrapSentenceSlider> {
  int _current = 0;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: MediaQuery.of(context).size.width,
          height: MediaQuery.of(context).size.height * 0.25,
          padding: const EdgeInsets.fromLTRB(0, 10, 0, 0),
          child: CarouselSlider(
            items: widget.scrapSentenceList.map((sentence) {
              return Builder(
                builder: (BuildContext context){
                  return scriptContentBlock(sentence, MediaQuery.of(context).size.width);
              });
            }).toList(),
            options: CarouselOptions(
              autoPlay: false,
              enableInfiniteScroll: false,
              viewportFraction: 1,
              aspectRatio: 2.0,
              onPageChanged: (idx, reason) {
                setState(() {
                  _current = idx;
                });
            })
          )
        ),
        DotsIndicator(
          dotsCount: widget.scrapSentenceList.length,
          position: _current,
          decorator: const DotsDecorator(
            color: colors.dotIndicatorColor,
            activeColor: colors.textColor,
          ),
        ),
      ]);
  }
}