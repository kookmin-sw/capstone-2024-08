import 'package:capstone/screen/script/script_list.dart';
import 'package:capstone/widget/tap_bar.dart';
import 'package:flutter/material.dart';

class ScriptTabs extends StatefulWidget{
  const ScriptTabs({Key? key}) : super(key: key);

  @override    
  ScriptTabsState createState() => ScriptTabsState();
}

class ScriptTabsState extends State<ScriptTabs> with SingleTickerProviderStateMixin {   
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
      appBar: listTapBar(_tabController, 'script'),
      body: TabBarView(
          controller: _tabController,
          children: [
            for(int idx=0; idx<_tabController.length; idx++)
              ScriptList(index: idx)
      ])
    );
  }
}