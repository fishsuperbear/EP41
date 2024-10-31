/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: desense_manager.cpp
 * @Date: 2023/12/20
 * @Author: kun
 * @Desc: --
 */

#include "processor/include/impl/desense_manager.h"
#include <regex>
#include <cmath>
#include "middleware/tools/data_tools/bag/include/reader.h"
#include "proto/soc/sensor_image.pb.h"
#include "utils/include/path_utils.h"
#include "utils/include/trans_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

std::mutex DesenseManager::m_mtx;
std::queue<DesenseNode> DesenseManager::m_desenseTaskQue;
std::map<std::string, TaskStatus> DesenseManager::m_name2Status;
std::map<std::string, std::vector<std::string>> DesenseManager::m_name2Output;

DesenseManager::DesenseManager() {
    m_taskStatus.store(TaskStatus::INITIAL, std::memory_order::memory_order_release);
    m_stopFlag = false;
}

DesenseManager::~DesenseManager() {}

void DesenseManager::onCondition(std::string type, char* data, Callback callback) {}

void DesenseManager::configure(std::string type, YAML::Node& node) {
    m_desenseManagerOption = node.as<DesenseManagerOption>();
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void DesenseManager::configure(std::string type, DataTrans& node) {
    m_taskStatus.store(TaskStatus::CONFIGURED, std::memory_order::memory_order_release);
}

void DesenseManager::desen() {
    while (m_stopFlag == false) {
        m_mtx.lock();
        if (m_desenseTaskQue.empty()) {
            m_mtx.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } else {
            DesenseNode desenNode = m_desenseTaskQue.front();
            m_desenseTaskQue.pop();
            m_mtx.unlock();

            if (m_desenseManagerOption.enable == false) {
                DC_SERVER_LOG_DEBUG << "desense manager disbale";
                m_name2Output[desenNode.taskName] = desenNode.inputFilePathVec;
                m_name2Status[desenNode.taskName] = TaskStatus::FINISHED;
                continue;
            }

            if (m_desenseManagerOption.outputFolderPath.empty()) {
                DC_SERVER_LOG_ERROR << "desense manager: output path missed";
                m_name2Output[desenNode.taskName] = desenNode.inputFilePathVec;
                m_name2Status[desenNode.taskName] = TaskStatus::FINISHED;
                continue;
            }

            std::string outputFolderPath = PathUtils::getFilePath(m_desenseManagerOption.outputFolderPath, TransUtils::stringTransFileName("%Y%m%d-%H%M%S"));
            m_outputFilePathVec.clear();
            bool is_find_I_frame[11] = {false};
            std::vector<std::unique_ptr<bag::Writer>> writer_vec;
            for (uint i = 0; i < desenNode.inputFilePathVec.size(); i++) {
                writer_vec.push_back(std::make_unique<bag::Writer>());
                std::string fileName = PathUtils::getFileName(desenNode.inputFilePathVec[i]);
                bag::WriterOptions mcapWriterOptions;
                mcapWriterOptions.use_time_suffix = false;
                mcapWriterOptions.output_file_name = PathUtils::getFilePath(outputFolderPath, fileName);
                bag::WriterCallback callback = std::bind(&DesenseManager::setOutputFilePath, this, std::placeholders::_1);
                writer_vec[i]->WriterRegisterCallback(callback);
                writer_vec[i]->Open(mcapWriterOptions);
            }
            for (int i = 0; i < 11; i++) {
                int camera_index;
                if (i < 3) {
                    camera_index = i;
                } else {
                    camera_index = i + 1;
                }
                DC_SERVER_LOG_DEBUG << "desense manager: camera_index " << camera_index;
                for (uint j = 0; j < desenNode.inputFilePathVec.size(); j++) {
                    std::unique_ptr<bag::Reader> reader;
                    reader = std::make_unique<bag::Reader>();
                    DC_SERVER_LOG_DEBUG << "desense manager: file path" << desenNode.inputFilePathVec[j];
                    reader->Open(desenNode.inputFilePathVec[j]);
                    while (reader->HasNext()) {
                        if (m_stopFlag) {
                            DC_SERVER_LOG_DEBUG << "desense manager: stop";
                            m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
                            return;
                        }
                        bag::TopicMessage msg = reader->ReadNext();
                        if (msg.data.empty()) {
                            DC_SERVER_LOG_ERROR << "desense manager: h265 msg data read error";
                            continue;
                        }
                        std::string camera_topic = "/soc/encoded_camera_" + std::to_string(camera_index);
                        if (msg.topic == camera_topic) {
                            auto inputImg = reader->DeserializeToProto<hozon::soc::CompressedImage>(msg);
                            // 1->I frame   2->P frame
                            if (!is_find_I_frame[i] && inputImg.frame_type() != 1) {
                                continue;
                            } else {
                                is_find_I_frame[i] = true;
                            }
                            if (inputImg.data().empty()) {
                                DC_SERVER_LOG_ERROR << "desense manager: h265 img data read error";
                                continue;
                            }
                            std::string input_buf = inputImg.data();
                            if (input_buf.empty()) {
                                DC_SERVER_LOG_ERROR << "desense manager: h265 string data read error";
                                continue;
                            }
                            std::string output_buff;
#ifdef BUILD_FOR_ORIN
                            m_desenProcessVec[i]->Process(input_buf, output_buff);
#else
                            output_buff = input_buf;
#endif
                            if (output_buff.empty()) {
                                DC_SERVER_LOG_ERROR << "desense manager: h265 data desen failed";
                                continue;
                            }
                            hozon::soc::CompressedImage outputImg;
                            outputImg.set_data(output_buff);
                            writer_vec[j]->WriteEventProtoMessage<hozon::soc::CompressedImage>(msg.topic, outputImg, msg.time);
                            std::this_thread::sleep_for(std::chrono::milliseconds(m_desenseManagerOption.delayMs));
                        }
                    }
                    reader->Close();
                }
            }

            // 删除脱敏前的数据源
            for (std::string filePath : desenNode.inputFilePathVec) {
                if (PathUtils::isFileExist(filePath)) {
                    PathUtils::removeFile(filePath);
                }
                std::string folderPath = PathUtils::getFolderName(filePath);
                std::smatch matchResult;
                std::tm tm{};
                std::string timeFormat = "\\d{4}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{6}";
                std::regex regexPattern = std::regex(timeFormat);
                if (std::regex_search(folderPath, matchResult, regexPattern)) {
                    if (PathUtils::isDirExist(folderPath)) {
                        PathUtils::removeFolder(folderPath);
                    }
                }
            }
            m_name2Output[desenNode.taskName] = m_outputFilePathVec;
            m_name2Status[desenNode.taskName] = TaskStatus::FINISHED;
        }
    }
}

void DesenseManager::active() {
    m_taskStatus.store(TaskStatus::RUNNING, std::memory_order::memory_order_release);
    DC_SERVER_LOG_DEBUG << "desense manager: begin";

#ifdef BUILD_FOR_ORIN
    for (int i = 0; i < 11; i++) {
        if (i < 2) {
            m_desenProcessVec.push_back(std::make_unique<desen::DesenProcess>(3840, 2160));
        } else {
            m_desenProcessVec.push_back(std::make_unique<desen::DesenProcess>(1920, 1080));
        }
    }
#endif

    m_th = std::make_unique<std::thread>(std::bind(&DesenseManager::desen, this));
}

void DesenseManager::deactive() {
    m_stopFlag = true;
    m_th->join();
    m_th.reset();
    DC_SERVER_LOG_DEBUG << "desense manager: end";
    m_taskStatus.store(TaskStatus::FINISHED, std::memory_order::memory_order_release);
}

TaskStatus DesenseManager::getStatus() {
    return m_taskStatus.load(std::memory_order::memory_order_acquire);
}

bool DesenseManager::getTaskResult(const std::string& taskName, struct DataTrans& dataStruct) {
    if (m_taskStatus.load(std::memory_order::memory_order_acquire) != TaskStatus::FINISHED) {
        return false;
    } else {
        return true;
    }
}

void DesenseManager::pause() {}

void DesenseManager::doWhenDestroy(const Callback& callback) {
    m_cb = callback;
}

void DesenseManager::setOutputFilePath(bag::WriterInfo& info) {
    std::string filePath = info.file_path;
    const std::string suffix = ".active";
    size_t suffixIndex = filePath.rfind(suffix);
    if (suffixIndex == filePath.size() - suffix.size()) {
        filePath = filePath.substr(0, suffixIndex);
    }
    m_outputFilePathVec.push_back(filePath);
}

bool DesenseManager::addDesenseTask(std::string taskName, std::vector<std::string> inputFilePathVec, int priority) {
    std::lock_guard<std::mutex> lg(m_mtx);
    if (m_name2Status.find(taskName) == m_name2Status.end()) {
        DesenseNode node;
        node.taskName = taskName;
        node.inputFilePathVec = inputFilePathVec;
        node.priority = priority;
        m_desenseTaskQue.push(node);
        m_name2Status[taskName] = TaskStatus::RUNNING;
        return true;
    } else {
        DC_SERVER_LOG_ERROR << "desense maneger: add failed, task is exist " << taskName;
        return false;
    }
}

bool DesenseManager::getDesenseTask(std::string taskName, std::vector<std::string>& outFilePathVec) {
    std::lock_guard<std::mutex> lg(m_mtx);
    if (m_name2Status.find(taskName) == m_name2Status.end()) {
        DC_SERVER_LOG_ERROR << "desense maneger: add failed, task not exist " << taskName;
        return false;
    } else {
        if (m_name2Status[taskName] == TaskStatus::FINISHED) {
            outFilePathVec = m_name2Output[taskName];
            m_name2Status.erase(taskName);
            m_name2Output.erase(taskName);
            return true;
        } else {
            return false;
        }
    }
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
