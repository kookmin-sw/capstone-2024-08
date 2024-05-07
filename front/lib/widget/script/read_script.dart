import 'package:capstone/model/script.dart';
import 'package:capstone/widget/script/script_list_tile.dart';
import 'package:flutter/material.dart'; 
 
 Widget readScripts(dynamic streamFunc, String scriptType){
    return StreamBuilder<List<ScriptModel>>(
        stream: streamFunc,
        builder: (context, snapshots) {
          if (snapshots.hasData) {
            return ListView.builder(
                itemCount: snapshots.data!.length,
                shrinkWrap: true,
                itemBuilder: (_, index) {
                  ScriptModel script = snapshots.data![index];
                  return scriptListTile(context, script, 'script', scriptType);
                });
        } else {
          return const SelectionArea(
            child: Center(
              child: Icon(Icons.hourglass_bottom)
            )
          );
        }
      })
    ;
  }