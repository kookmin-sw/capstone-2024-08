import 'package:capstone/screen/search/search_script.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;
import 'package:get/get.dart';

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

  IconButton moveToSearchPage() {
    return IconButton(
      icon: const Icon(
        Icons.search_rounded, 
        color: colors.blockColor
      ), 
      onPressed: () { Get.to(() => const SearchScript()); }
    );
  }

  PreferredSize listTapBar(TabController tabController, String route) {
    return PreferredSize(
      preferredSize: const Size.fromHeight(kToolbarHeight),
        child: Container(
          color: colors.bgrDarkColor,
          padding: const EdgeInsets.fromLTRB(0, 20, 10, 0),
          child: Row(
            children: [
              Expanded(
                child: TabBar(
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
                )
              ),
              route == 'script'
                ? moveToSearchPage()
                : Container()
          ])
        )
    );
  }        

  PreferredSize searchTapBar(TabController tabController) {
    return PreferredSize(
      preferredSize: const Size.fromHeight(kToolbarHeight),
      child: AppBar(
        backgroundColor: colors.bgrDarkColor,
        bottom: TabBar(
          controller: tabController,
          labelColor: colors.exampleScriptColor,
          unselectedLabelColor: colors.blockColor,
          dividerColor: colors.bgrDarkColor,
          indicatorColor: colors.exampleScriptColor,
          tabAlignment: TabAlignment.fill,
          tabs: _tabs.map((label) => Container(
              child: label,
          )).toList(),
        )
        )
    );
  }        

