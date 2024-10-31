/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: get_dynamic_config.h
 * @Date: 2023/12/5
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_GET_DYNAMIC_CONFIG_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_GET_DYNAMIC_CONFIG_H__

#include "include/yaml_cpp_struct.hpp"
#include "processor/include/processor.h"
#include "manager/include/config_manager.h"

namespace hozon {
namespace netaos {
namespace dc {

class GetDynamicConfig : public Processor {
public:
    GetDynamicConfig();
    ~GetDynamicConfig();
    void onCondition(std::string type, char* data, Callback callback) override;
    void configure(std::string type, YAML::Node& node) override;
    void configure(std::string type, DataTrans& node) override;
    void active() override;
    void deactive() override;
    TaskStatus getStatus() override;
    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override;
    void pause() override;
    void doWhenDestroy(const Callback& callback) override;
    void createYaml(Mapping_ mapping, std::string yamlFilePath);

private:
    std::atomic<TaskStatus> m_taskStatus;
    std::atomic_bool m_stopFlag;
    Callback m_cb;
    std::mutex m_mtx;
    TriggerIdPriority_ m_model;
    std::string m_cdnConfigFilePath;
    std::string m_oldTriggerDynamicConfigMD5;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_GET_DYNAMIC_CONFIG_H__
