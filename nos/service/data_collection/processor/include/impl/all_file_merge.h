/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: all_file_merge.h
 * @Date: 2023/11/23
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_ALL_FILE_MERGE_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_ALL_FILE_MERGE_H__

#include "include/yaml_cpp_struct.hpp"
#include "processor/include/processor.h"
#include "processor/include/impl/processor_impl_struct.h"

YCS_ADD_STRUCT(hozon::netaos::dc::AllFileMergeOption, outputFolderPath)

namespace hozon {
namespace netaos {
namespace dc {

class AllFileMerge : public Processor {
public:
    AllFileMerge();
    ~AllFileMerge();
    void onCondition(std::string type, char* data, Callback callback) override;
    void configure(std::string type, YAML::Node& node) override;
    void configure(std::string type, DataTrans& node) override;
    void active() override;
    void deactive() override;
    TaskStatus getStatus() override;
    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override;
    void pause() override;
    void doWhenDestroy(const Callback& callback) override;
    std::string changeFilePath(std::string oldFilePath, int index, std::string end);
    void deleteOldMcapFile(std::vector<std::string> mcapFilePathVec);
    void changeTriggerTime();
    void getUuid();

private:
    std::atomic<TaskStatus> m_taskStatus;
    std::atomic_bool m_stopFlag;
    Callback m_cb;
    std::mutex m_mtx;
    DataTrans m_inputNode;
    AllFileMergeOption m_allFileMergeOption;
    std::vector<std::string> m_outputMcapFilePathVec;
    std::string m_uuid;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_ALL_FILE_MERGE_H__
