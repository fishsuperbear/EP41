/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: collection_manager.h
 * @Date: 2023/07/13
 * @Author: cheng
 * @Desc: --
 */
#pragma once
#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_MANAGER_H_
#define MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_MANAGER_H_

#include <mutex>
#include <semaphore.h>
#include <vector>

#include "basic/trans_struct.h"
#include "collection_factory.h"
#include "manager/include/manager.h"
#include "thread_pool/include/thread_pool_flex.h"
#include "timer/timer_manager.hpp"
#include "utils/include/dc_logger.hpp"

namespace hozon {
namespace netaos {
namespace dc {

class CollectionManager : public Manager {
public:
    CollectionManager(ConfigManager *cfgm);
    CollectionManager(std::string type, ConfigManager *cfgm);
    ~CollectionManager();

    BasicTask* getTask(std::string taskType)  override{
        return CollectionFactory::getInstance()->createCollection( magic_enum::enum_cast<CollectionTypeEnum>(taskType).value());
    };
    std::string getTaskName(std::string taskType)  override{
        return taskType+TimeUtils::timestamp2ReadableStr(TimeUtils::getDataTimestamp());;
    };
    std::string runTaskb(YAML::Node &configureYaml, struct DataTrans&dataStruct);
private:
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_MANAGER_H_
