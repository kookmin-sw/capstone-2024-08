import 'package:capstone/widget/script/read_script.dart';
import 'package:capstone/widget/tap_bar.dart';
import 'package:flutter/material.dart';
import 'package:capstone/model/load_data.dart';
import 'package:capstone/constants/color.dart' as colors;

class SearchTabs extends StatefulWidget{
  const SearchTabs({
    Key? key,
    required this.query
  }) : super(key: key);

  final String? query;
  @override    
  SearchTabsState createState() => SearchTabsState();
}

class SearchTabsState extends State<SearchTabs> with SingleTickerProviderStateMixin {   
  final LoadData loadData = LoadData();
  late TabController _tabController;

  @override    
  void initState(){        
    super.initState();            
    _tabController = TabController(vsync: this, length: 2);    
    setState(() {});
  }        

  @override    
  void dispose(){        
    _tabController.dispose();        
    super.dispose();    
  }    

  @override
  Widget build(BuildContext context){
    return Scaffold(
      appBar: searchTapBar(_tabController),
      body: Container(
        color: colors.bgrDarkColor,
        padding: const EdgeInsets.only(left: 35),
        child: TabBarView(
          controller: _tabController,
          children: [
            widget.query == ''
              ? Container()
              : readScripts(loadData.searchExampleScript(widget.query), 'example'),
            widget.query == ''
              ? Container()
              : readScripts(loadData.searchUserScript(widget.query), 'user'),
      ])
    ));
  }
}