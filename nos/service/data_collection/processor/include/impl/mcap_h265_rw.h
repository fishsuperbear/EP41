/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcap_h265_rw.h
 * @Date: 2023/10/16
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_MCAP_H265_RW_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_MCAP_H265_RW_H__

#include <iostream>
#include <vector>
#include "yaml-cpp/yaml.h"
#include "processor/include/processor.h"

namespace hozon {
namespace netaos {
namespace dc {

class McapH265RW : public Processor {
public:
    McapH265RW();
    ~McapH265RW();
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
    DataTrans m_node;
    Callback m_cb;
    std::mutex m_mtx;
    std::vector<std::string> m_inputH265FilePathVec;
    std::vector<std::string> m_outputH265FilePathVec;
    static std::mutex m_mtxForGetTaskName;
    static int m_number;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_MCAP_H265_RW_H__
