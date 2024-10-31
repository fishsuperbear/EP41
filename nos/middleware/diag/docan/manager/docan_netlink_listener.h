/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanEventSender Header
 */

#ifndef DOCAN_NETLINK_LISTENER_H_
#define DOCAN_NETLINK_LISTENER_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <thread>
#include "diag/docan/taskbase/docan_task_event.h"

namespace hozon {
namespace netaos {
namespace diag {

#define DOCAN_NETLINK_BUFF_SIZE     (64 * 1024)

    class DocanNetlinkListener
    {
    public:
        static DocanNetlinkListener* instance();
        static void destory();

        int32_t         Init(void);
        int32_t         Start(void);
        int32_t         Stop(void);
        int32_t         Deinit(void);

        void            parse(uint8_t* buff, uint32_t size);

    private:
        DocanNetlinkListener();
        virtual ~DocanNetlinkListener();

        DocanNetlinkListener(const DocanNetlinkListener&);
        DocanNetlinkListener& operator=(const DocanNetlinkListener&);

        int32_t         PollThread();

        void            postEvent(DocanTaskEvent *ev);
        int32_t         setupNetlinkSocket();
        int32_t         recvNetlinkMessage();

    private:
        static          DocanNetlinkListener* s_instance;

        int32_t         m_sockfd;
        uint8_t         m_buff[DOCAN_NETLINK_BUFF_SIZE * 2];
        std::thread     m_pollThread;
        bool            m_stopFlag = false;

    };

} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_NETLINK_LISTENER_H_
/* EOF */
