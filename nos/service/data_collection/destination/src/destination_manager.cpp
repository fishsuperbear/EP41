/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: upload_manager.cpp
 * @Date: 2023/08/16
 * @Author: cheng
 * @Desc: --
 */

#include <filesystem>
#include <iostream>

#include "utils/include/path_utils.h"
//#include "destination/include/advc_upload.h"
#include "destination/include/destination_manager.h"

namespace hozon {
namespace netaos {
namespace dc {

DestinationManager::DestinationManager(ConfigManager *cfgm) : Manager(cfgm){
    DC_SERVER_LOG_INFO << "dm init end";
}
DestinationManager::DestinationManager(std::string type, ConfigManager *cfgm) : Manager(type,cfgm),threadPoolForAdvc_(4,7){
    DC_SERVER_LOG_INFO << "dm init end";
}
//DestinationManager::DestinationManager() : threadPoolFlex_(9), tm_(new TimerManager()), stopFlag_(false) {
//    DC_SERVER_LOG_INFO << "dm init start";
//    taskDemon_ = std::thread(
//        [this] {
////          Manager::checkRunTimeoutTasks(threadPoolFlex_, tm_, stopFlag_);
//        }
//    );
//    DC_SERVER_LOG_INFO << "dm init end";
//}


DestinationManager::~DestinationManager() {
    threadPoolForAdvc_.stop();
    threadPoolFlex_.stop();
}


}  // namespace dc
}  // namespace netaos
}  // namespace hozon
