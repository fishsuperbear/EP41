/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: task_factory.h
 * @Date: 2023/07/13
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_FACTORY_H_
#define MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_FACTORY_H_

#include "basic/basic_task.h"
#include "collection/include/collection.h"
#include "utils/include/dc_logger.hpp"

namespace hozon {
namespace netaos {
namespace dc {

enum CollectionTypeEnum {
    BAG_RECORDER,
    LOG_COLLECTOR,
    FIXED_FILES_COLLECTOR,
    ALL_LARGE_LOG_COLLECTOR,
    CYCLE_RECORDER,
    CAN_DATA,
    ETH_DATA,
    MCU_DATA,
    HZ_LOG,
    HZ_FAULT,
    MCU_BAG_RECORDER,
    MCU_BAG_COLLECTOR,
    MCU_LOG_COLLECTOR,
    CAN_BAG_COLLECTOR,
};

class CollectionFactory {
 public:
  static CollectionFactory *getInstance() {
    static CollectionFactory cf;
    return &cf;
  }

  Collection* createCollection(CollectionTypeEnum collectionType);
 private:
  CollectionFactory() {}
  CollectionFactory(CollectionFactory &&cf) = delete;
  ~CollectionFactory() {}
  CollectionFactory(const CollectionFactory &cf) = delete;
  CollectionFactory &operator=(const CollectionFactory &cf) = delete;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_DATA_COLLECT_COLLECTION_COLLECTION_FACTORY_H_
