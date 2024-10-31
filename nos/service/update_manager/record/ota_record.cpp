/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: log record
 */
#include "update_manager/record/ota_record.h"

#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include "json/json.h"

#include "update_manager/config/update_settings.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

const std::string ota_record_file_path = "/opt/usr/log/ota_log/ota_process_log.txt";

static std::map<uint32_t, std::string> MessageMap {
    {N_OK, "OK"},
    {N_ERROR, "N_ERROR"},
    {N_TIMEOUT_P2_CLIENT, "N_TIMEOUT_P2_CLIENT"},
    {N_TIMEOUT_P2START_CLIENT, "N_TIMEOUT_P2START_CLIENT"},
    {N_TIMEOUT_P3_CLIENT_PYH, "N_TIMEOUT_P3_CLIENT_PYH"},
    {N_TIMEOUT_P3_CLIENT_FUNC, "N_TIMEOUT_P3_CLIENT_FUNC"},
    {N_WRONG_SN, "N_WRONG_SN"},
    {N_UNEXP_PDU, "N_UNEXP_PDU"},
    {N_WFT_OVRN, "N_WFT_OVRN"},
    {N_BUFFER_OVFLW, "N_BUFFER_OVFLW"},
    {N_RX_ON, "N_RX_ON"},
    {N_WRONG_PARAMETER, "N_WRONG_PARAMETER"},
    {N_WRONG_VALUE, "N_WRONG_VALUE"},
    {N_USER_CANCEL, "N_USER_CANCEL"},
    {N_WAIT, "N_WAIT"},
    {N_RETRY_TIMES_LIMITED, "N_RETRY_TIMES_LIMITED"},
    {N_NRC, "N_NRC"},
};


OTARecoder::OTARecoder()
    : updating_process_flag_(false)
    , start_time_(0)
{
}

int32_t
OTARecoder::Init()
{
    UM_INFO << "OTARecoder::Init.";
    start_time_ = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    UM_INFO << "OTARecoder::Init Done.";
    return 0;
}

int32_t
OTARecoder::Deinit()
{
    UM_INFO << "OTARecoder::Deinit.";
    if (ofs_.is_open()) {
        ofs_.close();
    }
    UM_INFO << "OTARecoder::Deinit Done.";
    return 0;
}

int32_t OTARecoder::RestoreProgress()
{
    UPDATE_LOG_D("RestoreProgress");
    updating_process_flag_= true;
    timeMap_.clear();
    progressList_.clear();
    return 0;
}

int32_t OTARecoder::AddSensorProgress(const std::string& name, const std::string& version)
{
    UPDATE_LOG_D("AddSensorProgress name: %s.", name.c_str());
    UpdateProgress_t progress;
    progress.name = name;
    progress.status = 0;
    progress.progress = 0;
    progress.type = 1;   // 0: unknown, 1: sensor, 2: soc, 3: mcu
    progress.targetVersion = version;
    progressList_.push_back(progress);
    return progressList_.size();
}

int32_t OTARecoder::UpdateSensorProgressVersion(const std::string& name, const std::string& version)
{
    for (auto it : progressList_) {
        if (it.type == 1 && it.name == name) {
            UM_INFO << "sensor name is : " << name << " , version is : " << version;
            it.targetVersion = version;
            break;
        }
    }
    return 0;
}

int32_t OTARecoder::AddSocProgress(const std::string& name, const std::string& version)
{
    UPDATE_LOG_D("AddSocProgress name: %s.", name.c_str());
    UpdateProgress_t progress;
    progress.name = name;
    progress.status = 0;
    progress.progress = 0;
    progress.type = 2;   // 0: unknown, 1: sensor, 2: soc, 3: mcu
    progress.targetVersion = version;
    progressList_.push_back(progress);
    return progressList_.size();
}

uint8_t OTARecoder::GetSensorProgress(const std::string& name)
{
    uint8_t progress = 0;
    for (auto it : progressList_) {
        if (it.type == 1 && it.name == name) {
            progress = it.progress;
            break;
        }
    }
    return progress;
}

uint8_t OTARecoder::GetSocProgress(const std::string& name)
{
    uint8_t progress = 0;
    for (auto it : progressList_) {
        if (it.type == 2 && it.name == name) {
            progress = it.progress;
            break;
        }
    }
    return progress;
}

uint8_t OTARecoder::GetSensorTotalProgress()
{
    uint8_t cnt = 0;
    uint16_t progress = 0;
    if (progressList_.size() == 0) {
        return (uint8_t)(progress);
    }
    for (auto it : progressList_) {
        if (it.type == 1) {
            ++cnt;
            progress += it.progress;
        }
    }
    // 无sensor的case，会导致崩溃
    if (cnt == 0)
    {
        return 0;
    }
    return (uint8_t)(progress/cnt);
}
uint8_t OTARecoder::GetSocTotalProgress()
{
    uint8_t cnt = 0;
    uint16_t progress = 0;
    if (progressList_.size() == 0) {
        return (uint8_t)(progress);
    }
    for (auto it : progressList_) {
        if (it.type == 2) {
            ++cnt;
            progress += it.progress;
        }
    }
    // 无soc的case，会导致崩溃
    if (cnt == 0)
    {
        return 0;
    }
    return (uint8_t)(progress/cnt);
}

bool OTARecoder::IsSensorUpdateProcess()
{
    bool ret = false;
    for (auto it : progressList_) {
        if (it.type == 1) {
            // Sensor
            ret = true;
            break;
        }
    }
    return ret;
}

bool OTARecoder::IsSocUpdateProcess()
{
    bool ret = false;
    for (auto it : progressList_) {
        if (it.type == 2) {
            // SoC
            ret = true;
            break;
        }
    }
    return ret;
}

bool OTARecoder::IsUpdatingProcess()
{
    return updating_process_flag_;
}

void OTARecoder::SetActivateProcess()
{
    updating_process_flag_ = false;
}

bool OTARecoder::IsSensorUpdateCompleted()
{
    for (auto it : progressList_) {
        if (it.type == 1 && (it.status != 3 && it.status != 4)) {
            // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
            return false;
        }
    }
    return true;
}

bool OTARecoder::IsSensorUpdateCompleted(const std::vector<std::string>& name)
{
    for (auto name_it : name) {
        for (auto it : progressList_) {
            if (it.name == name_it) {
                if (it.type == 1 && (it.status != 3 && it.status != 4)) {
                    // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
                    return false;
                }
            }
        }
    }
    return true;
}

bool OTARecoder::IsSocUpdateCompleted()
{
    for (auto it : progressList_) {
        if (it.type == 2 && (it.status != 3 && it.status != 4)) {
            // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
            return false;
        }
    }
    return true;
}

bool OTARecoder::IsUpdateCompleted()
{
    if (ConfigManager::Instance().IsSensorUpdate() && !IsSensorUpdateCompleted()) {
        return false;
    }
    if (ConfigManager::Instance().IsSocUpdate() && !IsSocUpdateCompleted()) {
        return false;
    }
    return true;
}

void
OTARecoder::RecordUpdateVersion(std::string type, std::string curruentVersion, std::string targetVersion)
{
    if (!ofs_.is_open()) {
        ofs_ = std::ofstream(ota_record_file_path, std::ios::out | std::ios::app);
    }

    struct tm tm;
    time_t ts = time(0);
    localtime_r(&ts,&tm);
    char time_format[128];
    memset(time_format, '\0', sizeof(time_format));
    strftime(time_format, sizeof(time_format), "%Y-%m-%d %H:%M:%S", &tm);
    uint64_t local_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    uint64_t time_nsec = local_time % 1000000000;

    std::ostringstream oss;
    std::string typeStr = "[" + type + "]";
    std::string curVersionStr = "[Current Version: " + curruentVersion + "]";
    std::string tgtVersionStr = "[Target Version:" + targetVersion + "]\n";
    std::string time_str = std::string(time_format) + "." + std::to_string(time_nsec);
    oss << std::left << std::setw(35) << time_str << std::setw(25) << typeStr << std::setw(45) << curVersionStr << std::left << tgtVersionStr;

    ofs_.write(oss.str().c_str(), oss.str().size());
    ofs_.flush();
}

void
OTARecoder::RecordStart(std::string type, uint8_t progress)
{
    UPDATE_LOG_D("RecordStart type: %s, progress: %d.", type.c_str(), progress);
    if ("TOTAL" == type) {
        if (updating_process_flag_) {
            if (ofs_.is_open()) {
                ofs_.close();
            }
            ofs_ = std::ofstream(ota_record_file_path, std::ios::out | std::ios::trunc);
        }
        else {
            if (!ofs_.is_open()) {
                ofs_ = std::ofstream(ota_record_file_path, std::ios::out | std::ios::app);
            }
        }
    }

    struct tm tm;
    time_t ts = time(0);
    localtime_r(&ts,&tm);
    for (auto &it : progressList_) {
        if (it.name == type) {
            it.progress = progress;
            it.status = 1; // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
        }
    }

    std::string content;
    if (updating_process_flag_) {
        content = "Update Start!";
    }
    else {
        content = "Activate Start!";
    }

    char time_format[128];
    memset(time_format, '\0', sizeof(time_format));
    strftime(time_format, sizeof(time_format), "%Y-%m-%d %H:%M:%S", &tm);

    uint64_t start_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    timeMap_[type] = start_time;
    uint64_t time_nsec = start_time % 1000000000;

    std::ostringstream oss;
    std::string typeStr = "[" + type + "]";
    content = "[" + content + "]\n";
    std::string progressStr = "[" + std::to_string(progress) + "%]";
    std::string time_str = std::string(time_format) + "." + std::to_string(time_nsec);

    oss << std::left << std::setw(35) << time_str << std::setw(15) << typeStr << std::setw(10) << progressStr << std::left << content;
    ofs_.write(oss.str().c_str(), oss.str().size());
    ofs_.flush();
}

void
OTARecoder::RecordStepStart(std::string type, std::string step, uint8_t progress)
{
    if (!ofs_.is_open()) {
        UPDATE_LOG_D("RecordStepStart  !ofs_.is_open");
        ofs_ = std::ofstream(ota_record_file_path, std::ios::out | std::ios::app);
        timeMap_[type] = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    }

    struct tm tm;
    time_t ts = time(0);
    localtime_r(&ts,&tm);
    for (auto &it : progressList_) {
        if (it.name == type) {
            it.progress = progress;
            it.status = 2; // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
        }
    }

    if ("" == step) {
        return;
    }

    char time_format[128];
    memset(time_format, '\0', sizeof(time_format));
    strftime(time_format, sizeof(time_format), "%Y-%m-%d %H:%M:%S", &tm);
    uint64_t local_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    uint64_t time_nsec = local_time % 1000000000;

    std::string content = "START...";

    std::ostringstream oss;
    std::string typeStr = "[" + type + "]";
    std::string stepStr = "[" + step + "]";
    content = "[" + content + "]\n";
    std::string progressStr = "[" + std::to_string(progress) + "%]";
    std::string time_str = std::string(time_format) + "." + std::to_string(time_nsec);
    oss << std::left << std::setw(35) << time_str << std::setw(15) << typeStr << std::setw(10) << progressStr << std::setw(45) << stepStr << std::left << content;

    ofs_.write(oss.str().c_str(), oss.str().size());
    ofs_.flush();
}

void
OTARecoder::RecordStepFinish(std::string type, std::string step, uint32_t result, uint8_t progress)
{
    if (!ofs_.is_open()) {
        ofs_ = std::ofstream(ota_record_file_path, std::ios::out | std::ios::app);
    }

    struct tm tm;
    time_t ts = time(0);
    localtime_r(&ts,&tm);
    for (auto &it : progressList_) {
        if (it.name == type) {
            it.progress = progress;
            it.status = 2; // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
        }
    }

    if ("" == step) {
        return;
    }

    char time_format[128];
    memset(time_format, '\0', sizeof(time_format));
    strftime(time_format, sizeof(time_format), "%Y-%m-%d %H:%M:%S", &tm);
    uint64_t local_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    uint64_t time_nsec = local_time % 1000000000;

    std::ostringstream oss;
    std::string typeStr = "[" + type + "]";
    std::string stepStr = "[" + step + "]";
    std::string resultStr = "[" + MessageMap[result] + "]\n";
    std::string progressStr = "[" + std::to_string(progress) + "%]";
    std::string time_str = std::string(time_format) + "." + std::to_string(time_nsec);
    oss << std::left << std::setw(35) << time_str << std::setw(15) << typeStr << std::setw(10) << progressStr << std::setw(45) << stepStr << std::left << resultStr;

    ofs_.write(oss.str().c_str(), oss.str().size());
    ofs_.flush();
}

void
OTARecoder::RecordFinish(std::string type, uint32_t result, uint8_t progress)
{
    UPDATE_LOG_D("RecordFinish type: %s, result: %d, progress: %d.", type.c_str(), result, progress);
    if (!ofs_.is_open()) {
        ofs_ = std::ofstream(ota_record_file_path, std::ios::out | std::ios::app);
    }
    struct tm tm;
    time_t ts = time(0);
    localtime_r(&ts,&tm);
    for (auto &it : progressList_) {
        if (it.name == type) {
            it.progress = progress;
            // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
            it.status = (result == N_OK) ? 3 : 4;
        }
    }

    char time_format[128];
    memset(time_format, '\0', sizeof(time_format));
    strftime(time_format, sizeof(time_format), "%Y-%m-%d %H:%M:%S", &tm);
    uint64_t local_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    uint64_t time_nsec = local_time % 1000000000;

    uint64_t end_time = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    uint64_t diff = (start_time_ < timeMap_[type]) ? end_time - timeMap_[type] : end_time - start_time_;

    std::string content;
    if (updating_process_flag_) {
        content = "Update Completed! Cost(ms): " + std::to_string(diff/1000000);
    }
    else {
        content = "Activate Completed! Cost(ms): " + std::to_string(diff/1000000);
    }

    std::ostringstream oss;
    std::string typeStr = "[" + type + "]";
    content = "[" + content + "]";
    std::string resultStr = "[" + MessageMap[result] + "]\n";
    std::string progressStr = "[" + std::to_string(progress) + "%]";
    std::string time_str = std::string(time_format) + "." + std::to_string(time_nsec);
    oss << std::left << std::setw(35) << time_str << std::setw(15) << typeStr << std::setw(10) << progressStr << std::setw(45) << content << resultStr;

    ofs_.write(oss.str().c_str(), oss.str().size());
    ofs_.flush();

    if ("TOTAL" == type) {
        UPDATE_LOG_D("TOTAL updating_process_flag_: %d", updating_process_flag_);
        ofs_.close();
    }

}

}  // namespace update
}  // namespace netaos
}  // namespace hozon