import 'package:capstone/model/load_data.dart';
import 'package:capstone/model/record.dart';
import 'package:capstone/model/script.dart';
import 'package:capstone/widget/script/script_list_tile.dart';
import 'package:flutter/material.dart'; 
import 'package:multiple_stream_builder/multiple_stream_builder.dart';

LoadData loadData = LoadData();

Widget readRecordScripts(dynamic streamFunc, String scriptType){
    return StreamBuilder2<List<ScriptModel>, List<RecordModel>>(
        streams: StreamTuple2(streamFunc, loadData.readUserPracticeRecord(scriptType)),
        builder: (context, snapshots) {
          if(snapshots.snapshot1.hasData & snapshots.snapshot2.hasData){
            List<ScriptModel>? scriptList = snapshots.snapshot1.data;
            List<RecordModel>? recordList = snapshots.snapshot2.data;
              return ListView.builder(
                itemCount: recordList!.length,
                itemBuilder: (_, index) {
                  for(int idx=0; idx<scriptList!.length; idx++){
                    if(recordList[index].id == scriptList[idx].id) {
                      return scriptListTile(context, scriptList[idx], 'record', scriptType, record: recordList[index]);
                    }
                  }
                  return Container();
                }
              );
          } else { 
            return const SelectionArea(
              child: Center(
                child: Icon(Icons.hourglass_bottom)
              )
            );
          }
    });
}