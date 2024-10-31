/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanEventReceiver Header
 */

#ifndef DOCAN_EVENT_RECEIVER_H_
#define DOCAN_EVENT_RECEIVER_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif


#include "diag/docan/taskbase/docan_task_event.h"

namespace hozon {
namespace netaos {
namespace diag {

    class DocanTaskEvent;
    /**
     * @brief Class of DocanEventReceiver
     *
     * Docan service event handler.
     */
    class DocanEventReceiver
    {
    public:
        virtual ~DocanEventReceiver();
        /**
         * @brief singleton instance point
         */
        static DocanEventReceiver* instance();

        /**
         * @brief destory instance.
         */
        static void destroy();

        int32_t         Init(void);
        int32_t         Start(void);
        int32_t         Stop(void);
        int32_t         Deinit(void);

        int32_t         ReceiveCanPacket(const N_EcuInfo_t& ecuInfo, const CanPacket& packet);

        virtual void onEvent(DocanTaskEvent* ev);


    private:
        DocanEventReceiver();

        DocanEventReceiver(const DocanEventReceiver&);
        DocanEventReceiver& operator=(const DocanEventReceiver&);

        void postEvent(DocanTaskEvent *ev);

    private:
        static DocanEventReceiver* s_instance;

        std::vector<N_RouteInfo_t>  route_info_list_;

    };

} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_EVENT_RECEIVER_H_
/* EOF */
