/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: desense_manager.h
 * @Date: 2023/12/20
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_DESENSE_MANAGER_H__
#define SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_DESENSE_MANAGER_H__

#include <set>
#include <queue>
#include <thread>
#include "processor/include/processor.h"
#include "desen_process.h"
#include "middleware/tools/data_tools/bag/include/writer.h"
#include "processor/include/impl/processor_impl_struct.h"

YCS_ADD_STRUCT(hozon::netaos::dc::DesenseManagerOption, outputFolderPath, enable, delayMs)

namespace hozon {
namespace netaos {
namespace dc {

struct DesenseNode{
    std::string taskName;
    std::vector<std::string> inputFilePathVec;
    int priority;
};

class DesenseManager : public Processor {
public:
    DesenseManager();
    ~DesenseManager();
    void onCondition(std::string type, char* data, Callback callback) override;
    void configure(std::string type, YAML::Node& node) override;
    void configure(std::string type, DataTrans& node) override;
    void active() override;
    void deactive() override;
    TaskStatus getStatus() override;
    bool getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) override;
    void pause() override;
    void doWhenDestroy(const Callback& callback) override;
    void setOutputFilePath(bag::WriterInfo& info);
    void desen();
    static bool addDesenseTask(std::string taskName, std::vector<std::string> inputFilePathVec, int priority);
    static bool getDesenseTask(std::string taskName, std::vector<std::string>& outFilePathVec);

private:
    std::atomic<TaskStatus> m_taskStatus;
    std::atomic_bool m_stopFlag;
    DataTrans m_node;
    Callback m_cb;
    DesenseManagerOption m_desenseManagerOption;
#ifdef BUILD_FOR_ORIN
    std::vector<std::unique_ptr<desen::DesenProcess>> m_desenProcessVec; 
#endif
    std::vector<std::string> m_outputFilePathVec;
    std::unique_ptr<std::thread> m_th;
    static std::mutex m_mtx;
    static std::queue<DesenseNode> m_desenseTaskQue;
    static std::map<std::string, TaskStatus> m_name2Status;
    static std::map<std::string, std::vector<std::string>> m_name2Output;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_PROCESSOR_INCLUDE_IMPL_DESENSE_MANAGER_H__
