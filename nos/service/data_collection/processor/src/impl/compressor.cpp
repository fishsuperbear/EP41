/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: compressor.cpp
 * @Date: 2023/09/22
 * @Author: kun
 * @Desc: --
 */
#include "service/data_collection/processor/include/impl/compressor.h"
#include "include/magic_enum.hpp"
#include "utils/include/trans_utils.h"
#include "utils/include/path_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

Compressor::Compressor() {
    m_taskStatus.store(TaskStatus::INITIAL, std::memory_order::memory_order_release);
    m_stopFlag = false;
}

Compressor::~Compressor() {

}

void Compressor::onCondition(std::string type, char* data, Callback callback) {

}

void Compressor::configure(std::string type, YAML::Node& node) {
    m_compressOption = node.as<CompressOption>();
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void Compressor::configure(std::string type, DataTrans& node) {
    m_node = node;
    m_compressFileName = node.memoryDatas[uploadFileNameDefine];
    m_triggerId = node.memoryDatas[triggerIdData];
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void Compressor::active() {
    m_taskStatus.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    for (const auto& element : m_node.pathsList) {
        auto key = element.first;
        if (m_compressOption.notCompressTypes.find(std::string(magic_enum::enum_name(key)))!=m_compressOption.notCompressTypes.end()) {
            continue;
        }
        const std::set<std::string>& targetSet = element.second;
        m_pathList.insert(m_pathList.end(),targetSet.begin(),targetSet.end());
        m_node.pathsList[key].clear();
    }
    std::lock_guard<std::mutex> lg(m_mtx);
    int result = 0;
    if (m_compressOption.model == "compress") {
        if (m_compressOption.outputFileName != "none") {
            m_compressFileName = m_compressOption.outputFileName;
        }
        m_compressFileName = TransUtils::stringTransFileName(m_compressFileName, m_triggerId);
        switch (m_compressOption.compressType) {
        case ZIP:
        case TAR_GZ:
        case TAR_LZ4: {
            result = m_compress.compress_file(m_compressOption.outputFolderPath, m_compressFileName, m_pathList, m_compressOption.compressType, m_node.memoryDatas[uploadFileDeleteFlag]);
            break;
        }
        case GZ:
        case LZ4: {
            result = m_compress.compress_file(m_compressOption.outputFolderPath, m_compressFileName, m_pathList[0], m_compressOption.compressType);
            break;
        }
        default:
            m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
            return;
        }
    } else if (m_compressOption.model == "decompress") {
        result = m_compress.decompress_file(m_pathList[0], m_compressOption.outputFolderPath);
    } else {
        m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
        return;
    }
    if (result == 0) {
        if (m_node.memoryDatas[uploadFileDeleteFlag] == "1") {
            for (std::string path : m_pathList) {
                if (PathUtils::isFileExist(path)) {
                    PathUtils::removeFile(path);
                }
                std::string folderPath = PathUtils::getFolderName(path);
                if (PathUtils::isDirExist(folderPath)) {
                    PathUtils::fastRemoveFolder(folderPath, "/opt/usr/col/toupload/");
                }
            }
        }
        m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
    } else {
        m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
    }
}

void Compressor::deactive() {
    m_stopFlag = true;
}

TaskStatus Compressor::getStatus() {
    return m_taskStatus.load(std::memory_order::memory_order_acquire);
}

bool Compressor::getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) {
    if (m_taskStatus.load(std::memory_order::memory_order_acquire) != TaskStatus::FINISHED) {
        return false;
    } else {
        if (m_compressOption.model == "compress") {
            dataStruct.dataType = DataTransType::file;
        } else if (m_compressOption.model == "decompress") {
            dataStruct.dataType = DataTransType::folder;
        } else {
            m_taskStatus.store(TaskStatus::ERROR, std::memory_order::memory_order_release);
            return false;
        }
        dataStruct.memoryDatas[triggerIdData] = m_node.memoryDatas[triggerIdData];
        dataStruct.memoryDatas[triggerTime] = m_node.memoryDatas[triggerTime];
        dataStruct.pathsList[compressedFiles].insert(m_compress.get_result_path());
        return true;
    }
}

void Compressor::pause() {

}

void Compressor::doWhenDestroy(const Callback& callback) {
    m_cb = callback;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
