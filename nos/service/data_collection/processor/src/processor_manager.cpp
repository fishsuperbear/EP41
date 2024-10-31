/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: processor_manager.cpp
 * @Date: 2023/08/11
 * @Author: cheng
 * @Desc: --
 */


#include "processor/include/processor_manager.h"

namespace hozon {
namespace netaos {
namespace dc {

//ProcessorManager::ProcessorManager() : threadPoolFlex_(9), tm_(new TimerManager()), stopFlag_(false) {
//    DC_SERVER_LOG_INFO << "pm init start";
//
//    DC_NEW(tm_, TimerManager());
//    tm_->start(threadPoolFlex_);
//    DC_SERVER_LOG_INFO << "pm init end";
//}
ProcessorManager::ProcessorManager(ConfigManager *cfgm) : Manager(cfgm){
    DC_SERVER_LOG_INFO << "pm init end";
}
ProcessorManager::ProcessorManager(std::string type, ConfigManager *cfgm) : Manager(type, cfgm){
    DC_SERVER_LOG_INFO << "pm init end";
}


ProcessorManager::~ProcessorManager() {

}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
