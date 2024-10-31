/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: get_cdn_config.h
 * @Date: 2023/12/13
 * @Author: kun
 * @Desc: --
 */

#pragma once
#ifndef SERVICE_DATA_COLLECTION_DOWNLOAD_INCLUDE_DOWNLOAD_H__
#define SERVICE_DATA_COLLECTION_DOWNLOAD_INCLUDE_DOWNLOAD_H__

#include <map>

#include "common/timer/timer_manager.hpp"
#include "thread_pool/include/thread_pool_flex.h"
#include "utils/include/dc_logger.hpp"

namespace hozon {
namespace netaos {
namespace dc {

struct CdnStruct {
    std::string cdnPath;
    std::string carPath;
    std::string md5;
    std::string code;
};

struct TriggerStruct {
    std::string code;
    CdnStruct versionCdnStruct;
    CdnStruct modelFileCdnStruct;
    CdnStruct modelCfgCdnStruct;
};

struct MemoryTriggerStruct {
    std::string code;
    std::string versionMD5;
    std::string versionFilePath;
    std::string modelFileMD5;
    std::string modelFilePath;
    std::string modelCfgMD5;
    std::string modelCfgPath;
};

class Download {
public:
    Download();
    ~Download();
    void start();
    void stop();

private:
    void getCDNConfig();
    void downloadCdnFile(std::string triggerId, TriggerStruct triggerStruct);
    void downloadAndUpdate(MemoryTriggerStruct& memoryTriggerStruct, std::string triggerId, TriggerStruct triggerStruct);
    std::vector<MemoryTriggerStruct> m_memoryTriggerStruct;
    std::string m_triggerJsonMd5;
    std::shared_ptr<TimerManager> tm;
    std::shared_ptr<ThreadPoolFlex> m_threadPoolFlex;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_DOWNLOAD_INCLUDE_DOWNLOAD_H__
