/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanSysInterface Header
 */

#ifndef DOCAN_SYS_INTERFACE_H_
#define DOCAN_SYS_INTERFACE_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "diag/docan/socket/can_socket.h"
#include "diag/docan/socket/can_gateway.h"

namespace hozon {
namespace netaos {
namespace diag {


    /**
     * @brief Class of DocanSysInterface
     *
     * This class is a implement of DocanSysInterface.
     */
    class DocanSysInterface
    {
    public:
        static DocanSysInterface* instance();
        static void destroy();

        int32_t         Init(void);
        int32_t         Start(void);
        int32_t         Stop(void);
        int32_t         Deinit(void);

        int32_t AddCanPacket(uint16_t ecu, const CanPacket& packet);

        int32_t AddCanSendQueue(uint16_t ecu, const std::vector<CanPacket>& queue);

        int32_t AddAllCanPacket(const CanPacket& packet);

        int32_t AddAllCanSendQueue(const std::vector<CanPacket>& queue);

        int32_t StopChannel(uint16_t ecu);
        bool    IsChannelStopped(uint16_t ecu);

    private:
        static DocanSysInterface*    s_instance;

        DocanSysInterface();
        virtual ~DocanSysInterface();

        DocanSysInterface(const DocanSysInterface&);
        DocanSysInterface& operator=(const DocanSysInterface&);

        std::vector<std::shared_ptr<CanSocket>> socket_list_;
        std::vector<std::shared_ptr<CanGateway>> route_list_;
    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_SYS_INTERFACE_H_
/* EOF */
