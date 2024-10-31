/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: upload_manager.h
 * @Date: 2023/08/16
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_UPLOAD_UPLOAD_MANAGER_H_
#define MIDDLEWARE_TOOLS_DATA_COLLECT_UPLOAD_UPLOAD_MANAGER_H_

#include <mutex>
#include <semaphore.h>
#include <vector>

#include "basic/trans_struct.h"
#include "manager/include/manager.h"
#include "thread_pool/include/thread_pool_flex.h"
#include "timer/timer_manager.hpp"
#include "utils/include/dc_logger.hpp"
#include "destination/include/advc_upload.h"
#include "yaml-cpp/yaml.h"

namespace hozon {
namespace netaos {
namespace dc {

class DestinationManager : public Manager {
public:
 DestinationManager(ConfigManager *cfgm);
 DestinationManager(std::string type, ConfigManager *cfgm);
    ~DestinationManager();
 BasicTask* getTask(std::string taskType) override{
     AdvcUploadTask * advcUpload;
     DC_NEW(advcUpload, AdvcUploadTask());
     advcUpload->setThreadPool(&threadPoolForAdvc_);
     return advcUpload;
 };
 std::string getTaskName(std::string taskType) override{
     return taskType+TimeUtils::timestamp2ReadableStr(TimeUtils::getDataTimestamp());
 };
    bool getTaskResult(const std::string &taskName, struct DataTrans&dataStruct) override {
        return true;
    }
private:
 ThreadPoolFlex threadPoolForAdvc_;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_UPLOAD_UPLOAD_MANAGER_H_
