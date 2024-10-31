/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: trans_struct.h
 * @Date: 2023/08/11
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_TRANS_STRUCT_H
#define MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_TRANS_STRUCT_H

#include <string>
#include <vector>
#include <set>
#include <map>


namespace hozon {
namespace netaos {
namespace dc {

enum TransStructType {
  base,
  dataTrans,
};

struct Base {
  std::string type;
  uint32_t timestamp;
};

enum DataTransType {
    file = 1,
    folder = 2,
    fileAndFolder = 3,
    memory = 4,
    allTypes = 7,
    null = 0,
};

enum FilesInListType {
    faultManagerFiles,
    planningFiles,
    hzLogFiles,
    videoTopicMcapFiles,
    commonTopicMcapFiles,
    calibrationFiles,
    softwareVersionFiles,
    uploadReqOnlyFiles,
    compressedFiles,
    decompressedFiles,
};
enum MemoryDataType {
    hardwareVersionData,
    softwareVersionData,
    calibrationData,
    calibrationFileName,
    triggerIdData,
    priorityData,
    triggerTime,
    uploadFileNameDefine,
    uploadFileDeleteFlag,
    MCUBagRecorderPointer,
};

struct DataTrans : Base {
  DataTransType dataType{DataTransType::null};
  std::map<FilesInListType, std::set<std::string>> pathsList;
  std::map<MemoryDataType,std::string> memoryDatas;
  int priority{-1};
  void mergeDataStruct(DataTrans& src) {
      auto dtt = src.dataType | dataType;
      if (dtt > DataTransType::memory) {
          dataType = DataTransType::allTypes;
      } else {
          dataType = static_cast<DataTransType>(dtt);
      }
      for (const auto& element : src.pathsList) {
          auto key = element.first;
          const std::set<std::string>& targetSet = element.second;
          std::set<std::string>& thisPathSet = pathsList[key];
          thisPathSet.insert(targetSet.begin(), targetSet.end());
      }
      for (const auto& element : src.memoryDatas) {
          auto key = element.first;
          if (!memoryDatas[key].empty() && !element.second.empty()) {
              if (memoryDatas[key]!=element.second) {
                  memoryDatas[key]=memoryDatas[key]="|"+element.second;
              }
          } else if (!element.second.empty()){
              memoryDatas[key] = element.second;
          }
      }
  }
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_BASIC_TRANS_STRUCT_H
