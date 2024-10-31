/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcap_changer.h
 * @Date: 2023/09/25
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_MCAP_CHANGER_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_MCAP_CHANGER_H__

#include <iostream>
#include "yaml-cpp/yaml.h"
#include "processor/include/processor.h"
#include "middleware/tools/data_tools/mcap/include/filter.h"
#include "middleware/tools/data_tools/mcap/include/merge.h"
#include "middleware/tools/data_tools/mcap/include/split.h"
#include "include/yaml_cpp_struct.hpp"
#include "processor/include/impl/processor_impl_struct.h"

YCS_ADD_STRUCT(hozon::netaos::dc::FilterOption, method, whiteTopicVec, blackTopicVec, outputPath)
YCS_ADD_STRUCT(hozon::netaos::dc::MergeOption, method, attachmentFilePathVec, outputPath)
YCS_ADD_STRUCT(hozon::netaos::dc::SplitOption, method, attachmentFilePathVec, outputPath)

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos::mcap;

class McapChanger : public Processor {
public:
    McapChanger();
    ~McapChanger();
    void onCondition(std::string type, char* data, Callback callback) override;
    void configure(std::string type, YAML::Node& node) override;
    void configure(std::string type, DataTrans& node) override;
    void active() override;
    void deactive() override;
    TaskStatus getStatus() override;
    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override;
    void pause() override;
    void doWhenDestroy(const Callback& callback) override;

private:
    std::atomic<TaskStatus> m_taskStatus;
    std::atomic_bool m_stopFlag;
    Callback m_cb;
    std::mutex m_mtx;
    std::string m_methodType;
    Filter m_filter;
    Merge m_merge;
    Split m_split;
    FilterOption m_filterOption;
    MergeOption m_mergeOption;
    SplitOption m_splitOption;
    DataTrans m_node;
    std::set<std::string> m_inputMcapFilePathVec[2];
    std::set<std::string> m_outputFilePathVec[2];
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_MCAP_CHANGER_H__
