/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanEventSender Header
 */

#ifndef DOCAN_EVENT_SENDER_H_
#define DOCAN_EVENT_SENDER_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <memory>
#include <mutex>
#include <map>
#include <vector>

namespace hozon {
namespace netaos {
namespace diag {

    class DocanListener;
    /**
     * @brief DocanEventSender class
     *
     * Docan service configure infomation.
     */
    class DocanEventSender
    {
    public:

        /**
         * @brief singleton instance point
         */
        static DocanEventSender* instance();

        /**
         * @brief destory instance.
         */
        static void destroy();

        int32_t         Init(void);
        int32_t         Start(void);
        int32_t         Stop(void);
        int32_t         Deinit(void);

        int32_t sendEvent(uint32_t evtType, uint32_t evtId, int32_t evtArg1, int32_t evtArg2, const std::vector<uint8_t>& evtData);

        int32_t sendConfirm(uint16_t reqTa, uint16_t reqSa, uint32_t reqId, uint32_t result);
        int32_t sendUdsResponse(const std::string& who, uint16_t reqTa, uint16_t reqSa, uint32_t reqId, uint32_t result, const std::vector<uint8_t>& data);
        int32_t sendUdsIndication(uint16_t reqTa, uint16_t reqSa, const std::vector<uint8_t>& data);

        bool    isListenerRegistered(const std::string& who);
        bool    isListenerRegistered(const std::string& who, const std::shared_ptr<DocanListener>& listener);
        int32_t addListener(const std::string& who, const std::shared_ptr<DocanListener>& listener);
        int32_t removeListener(const std::string& who);
        int32_t removeListener(const std::shared_ptr<DocanListener>& listener);


    private:
        virtual ~DocanEventSender();
        DocanEventSender();

        DocanEventSender(const DocanEventSender&);
        DocanEventSender& operator=(const DocanEventSender&);

    private:
        mutable std::mutex        m_sync;

        static DocanEventSender *s_instance;
        static std::mutex s_instance_mutex;

        std::map<std::string, std::shared_ptr<DocanListener>>    m_callbackMap;

    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_TASK_SENDER_H_
/* EOF */
