import 'package:capstone/constants/color.dart' as colors;
import 'package:capstone/screen/home/home.dart';
import 'package:capstone/screen/record/record_taps.dart';
import 'package:capstone/screen/script/script_taps.dart';
import 'package:flutter/material.dart';


class BottomNavBar extends StatefulWidget {
  const BottomNavBar({super.key});

  @override
  State<BottomNavBar> createState() => _BottomNavBarState();
}

class _BottomNavBarState extends State<BottomNavBar> {
  int _selectedIndex = 1;
  static final List<Widget> _widgetOptions = <Widget>[
    const RecordTabs(),
    const Home(),
    const ScriptTabs(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: _widgetOptions.elementAt(_selectedIndex),
      ),
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        items: <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.save),
            label: '기록',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: '홈',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.record_voice_over),
            label: '스크립트',
          ),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: colors.textColor,
        onTap: _onItemTapped,
      ),
    );
  }
}
