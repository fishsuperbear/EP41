/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: collection_manager.cpp
 * @Date: 2023/07/13
 * @Author: cheng
 * @Desc: --
 */

#include "collection/include/collection_manager.h"
#include "utils/include/time_utils.h"

namespace hozon {
namespace netaos {
namespace dc {
CollectionManager::CollectionManager(ConfigManager *cfgm) : Manager(cfgm) {
        DC_SERVER_LOG_INFO << "cm init end";
}

CollectionManager::CollectionManager(std::string type, ConfigManager *cfgm) : Manager(type, cfgm) {
        DC_SERVER_LOG_INFO << "cm init end";
}


CollectionManager::~CollectionManager() {
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
