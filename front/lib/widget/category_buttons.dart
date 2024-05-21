import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;
import 'package:capstone/constants/fonts.dart' as fonts;
import 'package:flutter/material.dart';

class CategoryButtons extends StatefulWidget {
  CategoryButtons({super.key, required this.onCategorySelected});

  final List<String> category = texts.category;
  final ValueChanged<String> onCategorySelected;

  @override
  State<CategoryButtons> createState() => _CategoryButtonsState();
}

class _CategoryButtonsState extends State<CategoryButtons> {
  String selectedCategory = '';

  @override
  void initState() {
    super.initState();
    selectedCategory = widget.category[0];
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(children: [
          for (int i = 0; i < widget.category.length; i++)
            Container(
                padding: const EdgeInsets.fromLTRB(0, 0, 6, 0),
                child: ChoiceChip(
                  visualDensity:
                      const VisualDensity(horizontal: -2, vertical: -2),
                  showCheckmark: false,
                  label: Text(widget.category[i],
                      semanticsLabel: widget.category[i],
                      style: const TextStyle(
                        color: colors.textColor,
                        fontSize: fonts.category,
                        fontWeight: FontWeight.w500
                      )),
                  selected: selectedCategory == widget.category[i],
                  onSelected: (bool selected) {
                    setState(() {
                      if (selected) {
                        selectedCategory = widget.category[i];
                        widget.onCategorySelected(selectedCategory);
                      }
                    });
                  },
                  clipBehavior: Clip.antiAlias,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                  ),
                  side: const BorderSide(color: colors.blockColor),
                  selectedColor: colors.exampleScriptColor,
                  backgroundColor: colors.blockColor,
                ))
        ]));
  }
}
