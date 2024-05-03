import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

final List<Tab> _tabs = [
    const Tab(
      child: Text(
        'News',
        semanticsLabel: 'News',
        style: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w800
        )
      ),
    ),
    const Tab(
      child: Text(
        'User',
        semanticsLabel: 'User',
        style: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w800
        )
      )
    )
  ];

  PreferredSize tapBar(TabController tabController) {
    return PreferredSize(
      preferredSize: const Size.fromHeight(kToolbarHeight),
      child: AppBar(
        backgroundColor: colors.bgrDarkColor,
        bottom: TabBar(
          controller: tabController,
          labelColor: colors.exampleScriptColor,
          unselectedLabelColor: colors.blockColor,
          dividerColor: colors.bgrDarkColor,
          indicatorColor: Colors.transparent,
          isScrollable: true,
          tabAlignment: TabAlignment.start, 
          labelPadding: const EdgeInsets.only(left: 20),
          tabs: _tabs.map((label) => Container(
              child: label,
          )).toList(),
        ))
    );
  }        

