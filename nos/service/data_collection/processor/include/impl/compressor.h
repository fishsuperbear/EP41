/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: compressor.h
 * @Date: 2023/09/22
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_COMPRESSOR_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_COMPRESSOR_H__

#include <iostream>
#include "yaml-cpp/yaml.h"
#include "processor/include/processor.h"
#include "common/compress/include/dc_compress.h"
#include "include/yaml_cpp_struct.hpp"
#include "processor/include/impl/processor_impl_struct.h"

YCS_ADD_STRUCT(hozon::netaos::dc::CompressOption, model, compressType, outputFolderPath, outputFileName)

namespace hozon {
namespace netaos {
namespace dc {

class Compressor : public Processor {
public:
    Compressor();
    ~Compressor();
    void onCondition(std::string type, char* data, Callback callback) override;
    void configure(std::string type, YAML::Node& node) override;
    void configure(std::string type, DataTrans& node) override;
    void active() override;
    void deactive() override;
    TaskStatus getStatus() override;
    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override;
    void pause() override;
    void doWhenDestroy(const Callback& callback) override;

    void unlockAndSetTaskStatus(TaskStatus status);

private:
    std::atomic<TaskStatus> m_taskStatus;
    std::atomic_bool m_stopFlag;
    DataTrans m_node;
    Callback m_cb;
    std::mutex m_mtx;
    Compress m_compress;
    CompressOption m_compressOption;
    std::vector<std::string> m_pathList;
    std::string m_compressFileName;
    std::string m_resultPath;
    std::string m_triggerId;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_COMPRESSOR_H__
