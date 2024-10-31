#include <stdio.h>
#include <string.h>
#include <cinttypes>
#include <thread>
#include <fstream>
#include <sys/stat.h>
#include "system_monitor/include/monitor/system_monitor_base.h"
#include "system_monitor/include/handler/system_monitor_handler.h"

namespace hozon {
namespace netaos {
namespace system_monitor {

#define BACKUP_SUFFIX ".backup"
#define RECORD_FILE_MAX_SIZE 10 * 1024 * 1024

const std::vector<uint32_t> STMM_FAULT_LIST = {405001, 406001, 407001, 408001, 409001, 409002, 410001, 410002, 410003};

SystemMonitorBase::SystemMonitorBase(const SystemMonitorSubFunctionInfo& funcInfo, const std::string& fileName)
: sub_function_info_(funcInfo)
, record_file_name_(fileName)
, record_str_("")
, record_stop_flag_(false)
, record_size_(0)
{
    struct stat buffer;
    if (0 == stat((sub_function_info_.recordFilePath + record_file_name_).c_str(), &buffer)) {
        record_size_ = buffer.st_size;
    }

    for (auto& fault : STMM_FAULT_LIST) {
        fault_status_map_.insert(std::make_pair(fault, 0));
    }
}

SystemMonitorBase::~SystemMonitorBase()
{
    fault_status_map_.clear();
}

// monitor name, id
std::string
SystemMonitorBase::GetMonitorName()
{
    return sub_function_info_.shortName;
}

SystemMonitorSubFunctionId
SystemMonitorBase::GetMonitorId()
{
    return sub_function_info_.id;
}

// monitor switch
void
SystemMonitorBase::SetMonitorSwitch(const std::string& monitorSwitch)
{
    sub_function_info_.monitorSwitch = monitorSwitch;
}

std::string
SystemMonitorBase::GetMonitorSwitch()
{
    return sub_function_info_.monitorSwitch;
}

// monitor cycle
void
SystemMonitorBase::SetMonitorCycle(const uint32_t monitorCycle)
{
    sub_function_info_.monitorCycle = monitorCycle;
}

uint32_t
SystemMonitorBase::GetMonitorCycle()
{
    return sub_function_info_.monitorCycle;
}

// record file cycle
void
SystemMonitorBase::SetRecordFileCycle(const uint32_t recordFileCycle)
{
    sub_function_info_.recordFileCycle = recordFileCycle;
}

uint32_t
SystemMonitorBase::GetRecordFileCycle()
{
    return sub_function_info_.recordFileCycle;
}

// record file path
void
SystemMonitorBase::SetRecordFilePath(const std::string& recordFilePath)
{
    sub_function_info_.recordFilePath = recordFilePath;
}

std::string
SystemMonitorBase::GetRecordFilePath()
{
    return sub_function_info_.recordFilePath;
}

// is alarm
void
SystemMonitorBase::SetIsAlarm(const bool isAlarm)
{
    sub_function_info_.isAlarm = isAlarm;
}

bool
SystemMonitorBase::GetIsAlarm()
{
    return sub_function_info_.isAlarm;
}

// alarm value
void
SystemMonitorBase::SetAlarmValue(const uint8_t alarmValue)
{
    sub_function_info_.alarmValue = alarmValue;
}

uint8_t
SystemMonitorBase::GetAlarmValue()
{
    return sub_function_info_.alarmValue;
}

// post processing switch
void
SystemMonitorBase::SetPostProcessingSwitch(const std::string& postProcessingSwitch)
{
    sub_function_info_.postProcessingSwitch = postProcessingSwitch;
}

std::string
SystemMonitorBase::GetPostProcessingSwitch()
{
    return sub_function_info_.postProcessingSwitch;
}

void
SystemMonitorBase::StartRecord()
{
    record_stop_flag_ = false;
    std::thread stmm_base_record([this]() {
        while (!record_stop_flag_) {
            if ("" != record_str_) {
                WriteDataToFile();
            }

            std::this_thread::sleep_for(std::chrono::seconds(sub_function_info_.recordFileCycle));
        }
    });

    pthread_setname_np(stmm_base_record.native_handle(), "stmm_base_record");
    stmm_base_record.detach();
}

void
SystemMonitorBase::StopRecord()
{
    record_stop_flag_ = true;
}

void
SystemMonitorBase::SetRecordStr(const std::string& recordStr)
{
    record_str_ += recordStr + "\n";
}

void
SystemMonitorBase::WriteDataToFile(const bool overwrite)
{
    std::ofstream ofs;
    if (overwrite) {
        ofs.open((sub_function_info_.recordFilePath + record_file_name_), std::ios::out | std::ios::binary);
    }
    else {
        ofs.open((sub_function_info_.recordFilePath + record_file_name_), std::ios::out | std::ios::app | std::ios::binary);
    }

    if (ofs.is_open()) {
        ofs << GetCurrentTime();
        ofs << record_str_;
        record_size_ += record_str_.size();
        record_str_ = "";
        ofs.close();
        if (record_size_ >= RECORD_FILE_MAX_SIZE) {
            int result = rename((sub_function_info_.recordFilePath + record_file_name_).c_str(), (sub_function_info_.recordFilePath + record_file_name_ + BACKUP_SUFFIX).c_str());
            if (-1 != result) {
                record_size_ = 0;
            }
        }
    }
}

void
SystemMonitorBase::RefreshFile(const std::string& reason)
{
    if ("" != record_str_) {
        record_str_ = reason + "\n" + record_str_;
        WriteDataToFile();
    }
}

bool
SystemMonitorBase::CopyFile(const std::string& from, const std::string& to)
{
    std::ifstream input(from, std::ios::binary);
    if (!input.good()) {
        return false;
    }

    std::ofstream output(to, std::ios::binary);
    if (!output.good()) {
        input.close();
        return false;
    }

    output << input.rdbuf();
    input.close();
    output.close();
    return true;
}

void
SystemMonitorBase::Notify(const std::string info)
{
    // SystemMonitorHandler::getInstance()->NotifyEventSend(sub_function_info_.id, info);
}

void
SystemMonitorBase::Alarm(const std::string info)
{
    // if (sub_function_info_.isAlarm) {
    //     SystemMonitorHandler::getInstance()->AlarmEventSend(sub_function_info_.id, info);
    // }
}

void
SystemMonitorBase::ReportFault(const uint32_t fault, const uint8_t faultStatus)
{
    SystemMonitorSendFaultInfo faultInfo;
    faultInfo.faultId = fault / 100;
    faultInfo.faultObj = fault % 100;
    faultInfo.faultStatus = faultStatus;
    auto itr = fault_status_map_.find(fault);
    if (itr != fault_status_map_.end()) {
        if (faultStatus != itr->second) {
            if (SystemMonitorHandler::getInstance()->ReportFault(faultInfo)) {
                fault_status_map_[fault] = faultStatus;
            }
        }
    }
}

void
SystemMonitorBase::Control(const SystemMonitorSubFunctionControlType& controlType, const std::string controlValue)
{
    switch(controlType)
    {
        case SystemMonitorSubFunctionControlType::kMonitorSwitch:
            if (controlValue != sub_function_info_.monitorSwitch) {
                if ("on" == controlValue) {
                    Start();
                }
                else {
                    Stop();
                }

                sub_function_info_.monitorSwitch = controlValue;
            }

            break;
        case SystemMonitorSubFunctionControlType::kMonitorCycle: {
            uint32_t monitorCycle = static_cast<uint32_t>(std::strtoul(controlValue.c_str(), 0, 10));
            if (monitorCycle != sub_function_info_.monitorCycle) {
                sub_function_info_.monitorCycle = monitorCycle;
            }

            break;
        }
        case SystemMonitorSubFunctionControlType::kRecordFileCycle: {
            uint32_t recordFileCycle = static_cast<uint32_t>(std::strtoul(controlValue.c_str(), 0, 10));
            if (recordFileCycle != sub_function_info_.recordFileCycle) {
                if (0 == recordFileCycle) {
                    record_stop_flag_ = true;
                }
                else {
                    if (0 == sub_function_info_.recordFileCycle) {
                        sub_function_info_.recordFileCycle = recordFileCycle;
                        StartRecord();
                    }
                }

                sub_function_info_.recordFileCycle = recordFileCycle;
            }

            break;
        }
        case SystemMonitorSubFunctionControlType::kRecordFilePath:
            if (("" != controlValue) && (controlValue != sub_function_info_.recordFilePath)) {
                sub_function_info_.recordFilePath = controlValue;
            }

            break;
        case SystemMonitorSubFunctionControlType::kIsAlarm: {
            bool isAlarm = static_cast<bool>(std::strtoul(controlValue.c_str(), 0, 10));
            if (isAlarm != sub_function_info_.isAlarm) {
                sub_function_info_.isAlarm = isAlarm;
            }

            break;
        }
        case SystemMonitorSubFunctionControlType::kAlarmValue: {
            uint8_t alarmValue = static_cast<uint8_t>(std::strtoul(controlValue.c_str(), 0, 10));
            if (alarmValue != sub_function_info_.alarmValue) {
                sub_function_info_.alarmValue = alarmValue;
            }

            break;
        }
        case SystemMonitorSubFunctionControlType::kPostProcessingSwitch:
            if (("" != controlValue) && (controlValue != sub_function_info_.postProcessingSwitch)) {
                sub_function_info_.postProcessingSwitch = controlValue;
            }

            break;
        default:
            break;
    }
}

std::string
SystemMonitorBase::GetCurrentTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y/%m/%d %H:%M:%S", localtime(&timep));
    std::string recordTime = "Record time: " + std::string(tmp) + "\n";
    return recordTime;
}

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon