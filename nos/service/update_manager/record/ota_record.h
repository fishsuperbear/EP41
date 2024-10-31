/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: log record
 */
#ifndef LOG_RECORD_H
#define LOG_RECORD_H

#include <fstream>
#include <string>
#include <unordered_map>
#include <list>
#include <vector>
#include <map>

namespace hozon {
namespace netaos {
namespace update {

typedef struct UpdateProgress {
    std::string name;
    uint8_t type;       // 0: unknown, 1: sensor, 2: soc, 3: mcu
    uint8_t status;     // 0: not start, 1: start 2: in progress, 3: completed OK, 4: completed failed
    uint8_t progress;   // progress 0 ~ 100
    std::string targetVersion;
} UpdateProgress_t;


class OTARecoder {
 public:
    static OTARecoder &Instance()
    {
        static OTARecoder instance;
        return instance;
    }

    int32_t Init();
    int32_t Deinit();

    int32_t RestoreProgress();

    int32_t AddSensorProgress(const std::string& name, const std::string& version);
    int32_t UpdateSensorProgressVersion(const std::string& name, const std::string& version);
    int32_t AddSocProgress(const std::string& name, const std::string& version);

    uint8_t GetSensorProgress(const std::string& name);
    uint8_t GetSocProgress(const std::string& name);

    uint8_t GetSensorTotalProgress();
    uint8_t GetSocTotalProgress();

    bool IsSensorUpdateProcess();
    bool IsSocUpdateProcess();
    bool IsUpdatingProcess();
    void SetActivateProcess();

    bool IsSensorUpdateCompleted();
    bool IsSensorUpdateCompleted(const std::vector<std::string>& name);
    bool IsSocUpdateCompleted();
    bool IsUpdateCompleted();

    void RecordUpdateVersion(std::string type, std::string curruentVersion, std::string targetVersion);
    void RecordStart(std::string type, uint8_t progress);
    void RecordStepStart(std::string type, std::string step, uint8_t progress);
    void RecordStepFinish(std::string type, std::string step, uint32_t result, uint8_t progress);
    void RecordFinish(std::string type, uint32_t result, uint8_t progress);

 private:
    OTARecoder();
    OTARecoder(const OTARecoder &);
    OTARecoder & operator = (const OTARecoder &);

    std::ofstream ofs_;
    std::unordered_map<std::string, uint64_t> timeMap_;
    std::list<UpdateProgress_t> progressList_;
    bool updating_process_flag_;
    uint64_t start_time_;

};
}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // LOG_RECORD_H