/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: add_data.h
 * @Date: 2023/11/13
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_ADD_DATA_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_ADD_DATA_H__

#include "yaml-cpp/yaml.h"
#include "include/yaml_cpp_struct.hpp"
#include "processor/include/processor.h"
#include "processor/include/impl/processor_impl_struct.h"

YCS_ADD_STRUCT(hozon::netaos::dc::AddDataOption, cmd, file, calibParamsVec)

namespace hozon {
namespace netaos {
namespace dc {

class AddData : public Processor {
public:
    AddData();
    ~AddData();
    void onCondition(std::string type, char* data, Callback callback) override;
    void configure(std::string type, YAML::Node& node) override;
    void configure(std::string type, DataTrans& node) override;
    void active() override;
    void deactive() override;
    TaskStatus getStatus() override;
    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override;
    void pause() override;
    void doWhenDestroy(const Callback& callback) override;
    std::string getCmdResult(const std::string &strCmd);

private:
    std::atomic<TaskStatus> m_taskStatus;
    std::atomic_bool m_stopFlag;
    Callback m_cb;
    std::mutex m_mtx;
    AddDataOption m_addDataOption;
    DataTrans m_inputNode;
    DataTrans m_outputNode;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_ADD_DATA_H__
