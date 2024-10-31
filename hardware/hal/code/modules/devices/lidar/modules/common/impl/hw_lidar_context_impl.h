#ifndef HW_LIDAR_CONTEXT_IMPL_H
#define HW_LIDAR_CONTEXT_IMPL_H

#include "lidar/modules/common/hw_lidar_modules_common.h"

#include "lidar/modules/common/impl/config/config_manager.h"
#include "lidar/modules/common/impl/protocol/protocol_socket.h"
#include "lidar/modules/common/impl/parser/parser_base.h"
#include "lidar/modules/common/impl/utils/lidar_types.h"
#include "lidar/modules/common/impl/utils/blocking_queue.h"

class HWLidarContext
{
public:
    HWLidarContext(struct hw_lidar_t *i_plidar) {}
    virtual ~HWLidarContext() {}

    virtual s32 Init() = 0;
    virtual s32 Device_Open(struct hw_lidar_callback_t *i_callback) = 0;
    virtual s32 Device_Close() = 0;
};

#endif // HW_LIDAR_CONTEXT_IMPL_H
