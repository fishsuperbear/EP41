/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: add_data.cpp
 * @Date: 2023/11/13
 * @Author: kun
 * @Desc: --
 */

#include <fstream>
#include <sstream>
#include "processor/include/impl/add_data.h"
#include "config_param.h"
#include "utils/include/path_utils.h"
#include "utils/include/trans_utils.h"
#include "json/json.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos::cfg;

AddData::AddData() {
    m_taskStatus.store(TaskStatus::INITIAL, std::memory_order::memory_order_release);
    m_stopFlag = false;
}

AddData::~AddData() {}

void AddData::onCondition(std::string type, char* data, Callback callback) {}

void AddData::configure(std::string type, YAML::Node& node) {
    m_addDataOption = node.as<AddDataOption>();
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void AddData::configure(std::string type, DataTrans& node) {
    m_inputNode = node;
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void AddData::active() {
    m_taskStatus.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    DC_SERVER_LOG_DEBUG << "add data: begin";
    std::lock_guard<std::mutex> lg(m_mtx);

    // 获取cmd命令的执行结果
    for (auto cmd : m_addDataOption.cmd) {
        if (m_stopFlag) {
            DC_SERVER_LOG_DEBUG << "add data: stop";
            m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
            return;
        }
        if (cmd.first.as<std::string>() == "devm_version") {
            std::string cmdResult = getCmdResult(cmd.second.as<std::string>());
            if (cmdResult.empty()) {
                DC_SERVER_LOG_WARN << "add data: cmd error " << cmd.first.as<std::string>();
            } else {
                m_outputNode.memoryDatas[hardwareVersionData] = cmdResult;
            }
        }
    }

    // 获取文件内容
    for (auto file : m_addDataOption.file) {
        if (m_stopFlag) {
            DC_SERVER_LOG_DEBUG << "add data: stop";
            m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
            return;
        }
        if (file.first.as<std::string>() == "version") {
            std::string filePath = file.second.as<std::string>();
            if (PathUtils::isFileExist(filePath)) {
                std::ifstream fileStream(filePath, std::ios::in | std::ios::binary);
                std::stringstream fileSStr;
                fileSStr << fileStream.rdbuf();
                fileStream.close();
                m_outputNode.memoryDatas[softwareVersionData] = fileSStr.str();
            } else {
                DC_SERVER_LOG_WARN << "add data: file not exist " << filePath;
            }
        }
    }

    // 获取标定参数
    if (BasicTask::collectFlag.calibrationCollect == true) {
        if (!m_addDataOption.calibParamsVec.empty()) {
            Json::Value kv_vec(Json::arrayValue);
            auto cfgMgr = ConfigParam::Instance();
            cfgMgr->Init();
            for (std::string cabliParam : m_addDataOption.calibParamsVec) {
                if (m_stopFlag) {
                    DC_SERVER_LOG_DEBUG << "add data: stop";
                    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
                    return;
                }
                std::string value;
                uint res = cfgMgr->GetParam<std::string>(cabliParam, value);
                if (res == 1) {
                    Json::Value data;
                    data["key"] = cabliParam;
                    data["value"] = value;
                    kv_vec.append(data);
                } else {
                    DC_SERVER_LOG_DEBUG << "add data: calibration param missed, key: " << cabliParam;
                }
            }
            if (kv_vec.empty()) {
                DC_SERVER_LOG_DEBUG << "add data: no calibration param";
            } else {
                Json::Value calibrationParamsJson;
                calibrationParamsJson["kv_vec"] = kv_vec;
                Json::StyledWriter writer;
                std::string calibParamsData = writer.write(calibrationParamsJson);
                m_outputNode.memoryDatas[MemoryDataType::calibrationData] = calibParamsData;
            }
        }
    } else {
    DC_SERVER_LOG_DEBUG << "add data: calibration collect flag is false";
    }
    
    DC_SERVER_LOG_DEBUG << "add data: end";
    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
}

void AddData::deactive() {
    m_stopFlag = true;
}

TaskStatus AddData::getStatus() {
    return m_taskStatus.load(std::memory_order::memory_order_acquire);
}

bool AddData::getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) {
    if (m_taskStatus.load(std::memory_order::memory_order_acquire) != TaskStatus::FINISHED) {
        return false;
    } else {
        dataStruct = m_outputNode;
        dataStruct.memoryDatas[triggerIdData] = m_inputNode.memoryDatas[triggerIdData];
        dataStruct.memoryDatas[triggerTime] = m_inputNode.memoryDatas[triggerTime];
        dataStruct.dataType = DataTransType::memory;
        dataStruct.mergeDataStruct(m_inputNode);
        return true;
    }
}

void AddData::pause() {}

void AddData::doWhenDestroy(const Callback& callback) {
    m_cb = callback;
}

/// 执行cmd指令并返回结果
std::string AddData::getCmdResult(const std::string &strCmd)  
{
    char buf[10240] = {0};
    FILE *pf = NULL;
 
    if( (pf = popen(strCmd.c_str(), "r")) == NULL )
    {
        return "";
    }
 
    std::string strResult;
    while(fgets(buf, sizeof buf, pf))
    {
        strResult += buf;
    }
 
    pclose(pf);
 
    unsigned int iSize =  strResult.size();
    if(iSize > 0 && strResult[iSize - 1] == '\n')  // linux
    {
        strResult = strResult.substr(0, iSize - 1);
    }
 
    return strResult;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
