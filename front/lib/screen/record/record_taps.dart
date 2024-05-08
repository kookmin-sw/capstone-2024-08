import 'package:capstone/screen/record/record_list.dart';
import 'package:capstone/widget/tap_bar.dart';
import 'package:flutter/material.dart';

class RecordTabs extends StatefulWidget{
  const RecordTabs({Key? key}) : super(key: key);

  @override    
  RecordTabsState createState() => RecordTabsState();
}

class RecordTabsState extends State<RecordTabs> with SingleTickerProviderStateMixin {   
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
      appBar: listTapBar(_tabController, 'record'),
      body: TabBarView(
        controller: _tabController,
        children: [
          for(int idx=0; idx<_tabController.length; idx++)
            RecordList(index: idx)
      ])
    );
  }
}