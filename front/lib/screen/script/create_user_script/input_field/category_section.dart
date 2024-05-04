import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/constants/text.dart' as texts;

class CategorySection extends StatefulWidget {
  const CategorySection({Key? key, required this.onCategorySelected})
      : super(key: key);

  final ValueChanged<String> onCategorySelected;

  @override
  _CategorySectionState createState() =>
      _CategorySectionState();
}

class _CategorySectionState extends State<CategorySection> {
  String _selectedCategory = '';

  @override
  void initState() {
    super.initState();
    _selectedCategory = '';
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(30, 10, 30, 30),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const Text(
            '카테고리를 선택해주세요.',
            style: TextStyle(
              color: colors.textColor
            ),
          ),
          Container(
              padding: const EdgeInsets.all(5.0),
              alignment: Alignment.center,
              child: Column(
                children: [
                for (int paragraph = 0; paragraph < 2; paragraph++)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center, 
                    children: [
                      for (int index = 3 * paragraph + 1;
                        index < 3 * paragraph + 4;
                        index++)
                      Container(
                          padding: const EdgeInsets.fromLTRB(5, 5, 5, 0),
                          child: ChoiceChip(
                            label: Text(texts.category[index],
                                style: const TextStyle(
                                    color: colors.textColor
                                )),
                            selected: _selectedCategory ==
                                texts.category[index],
                            onSelected: (bool selected) {
                              setState(() {
                                if (selected) {
                                  _selectedCategory = texts.category[index];
                                  widget.onCategorySelected(_selectedCategory);
                                }
                              });
                            },
                            clipBehavior: Clip.antiAlias,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                              side: const BorderSide(width: 0)
                            ),
                            selectedColor: colors.exampleScriptColor,
                            backgroundColor: colors.blockColor,
                            disabledColor: colors.blockColor,
                            showCheckmark: false,
                          ))
                  ])
              ]))
        ]));
  }
}
