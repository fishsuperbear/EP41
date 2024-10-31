/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: collection_factory.cpp
 * @Date: 2023/07/13
 * @Author: cheng
 * @Desc: --
 */

#include "collection/include/collection_factory.h"
#include <cstdio>
#include "collection/include/impl/bag_record.h"
#include "collection/include/impl/fixed_files_collector.h"
#include "collection/include/impl/log_collector.h"
#include "collection/include/impl/all_log_collector.h"
#include "collection/include/impl/mcu_bag_recorder.h"
#include "collection/include/impl/mcu_bag_collector.h"
#include "collection/include/impl/mcu_log_collector.h"
#include "collection/include/impl/can_bag_collector.h"

namespace hozon {
namespace netaos {
namespace dc {

Collection* CollectionFactory::createCollection(CollectionTypeEnum collectionType) {
    DC_SERVER_LOG_DEBUG<<"for debug";
    switch (collectionType) {
        case BAG_RECORDER: {
            DC_SERVER_LOG_DEBUG<<"for debug create";
            DC_RETURN_NEW(BagRecorder());
        };
        case LOG_COLLECTOR: {
            DC_SERVER_LOG_DEBUG<<"for debug";
            DC_RETURN_NEW(LogCollector());
        };
        case FIXED_FILES_COLLECTOR: {
            DC_SERVER_LOG_DEBUG<<"for debug";
            DC_RETURN_NEW(FixedFilesCollector());
        };
        case MCU_BAG_RECORDER: {
            DC_SERVER_LOG_DEBUG<<"MCU_BAG_RECORDER";
            DC_RETURN_NEW(MCUBagRecorder);
        };
        case MCU_BAG_COLLECTOR: {
            DC_SERVER_LOG_DEBUG<<"MCU_BAG_COLLECTOR";
            DC_RETURN_NEW(MCUBagCollector);
        }
        case MCU_LOG_COLLECTOR: {
            DC_SERVER_LOG_DEBUG<<"MCU_LOG_COLLECTOR";
            DC_RETURN_NEW(MCULogCollector);
        }
        case CAN_BAG_COLLECTOR: {
            DC_SERVER_LOG_DEBUG<<"CAN_BAG_COLLECTOR";
            DC_RETURN_NEW(CANBagCollector);
        }
        case ALL_LARGE_LOG_COLLECTOR: {
            DC_SERVER_LOG_DEBUG<<"ALL_LARGE_LOG_COLLECTOR";
            DC_RETURN_NEW(AllLogCollector());
        }
        default: {
            printf("++++++ create collection error: not support type %d\n", (int)collectionType);
            DC_SERVER_LOG_ERROR << "create collection error: not support type " << collectionType;
            return nullptr;
        };
    }
    DC_SERVER_LOG_ERROR << "create collection error: not support type " << collectionType;
    return nullptr;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
