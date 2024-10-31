/*!
 * @file diag_server_event_mgr.h
 * This file contains the implementation of the diag event manager functions.
 */

#pragma once

#include "diag/common/include/thread_pool.h"
#include "diag/diag_server/include/common/diag_server_def.h"
#include <iostream>
#include <vector>
#include <mutex>
#include <memory>

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerEventHandler
{
public:
    DiagServerEventHandler();
    virtual ~DiagServerEventHandler();
    DiagServerEventHandler(DiagServerEventHandler&) = delete;
    DiagServerEventHandler& operator= (const DiagServerEventHandler&) = delete;
    static DiagServerEventHandler* getInstance();

    // 14
    void clearDTCInformation(const uint32_t dtcGroup);

    // 19 01
    void reportNumberOfDTCByStatusMask(uint8_t dtcStatusMask);

    // 19 02
    void reportDTCByStatusMask(uint8_t dtcStatusMask);

    // 19 03
    void reportDTCSnapshotIdentification();

    // 19 04 123456(DTC) 01(number)
    void reportDTCSnapshotRecordByDTCNumber(const uint32_t dtcValue, const uint32_t number);

    // 19 0A
    void reportSupportedDTC();

    // 85
    void controlDTCStatusType(const DIAG_CONTROLDTCSTATUSTYPE& controlDtcStatusType);


    // reply 14
    void replyClearAllDtc(const bool result);

    // reply 19 01
    void replyNumberOfDTCByStatusMask(const uint32_t number);

    // reply 19 02
    void replyDTCByStatusMask(std::vector<DiagDtcData>& dtcInfos);

    // reply 19 03
    void replyDTCSnapshotIdentification(std::vector<DiagDtcData>& dtcInfos);

    // reply 19 04
    void replyDTCSnapshotRecordByDTCNumber(std::vector<DiagDtcData>& dtcInfos, uint8_t m_number);

    // reply 19 0A
    void replySupportedDTC(std::vector<DiagDtcData>& dtcInfos);

    // reply 85
    void replyControlDTCStatusType(const DIAG_CONTROLDTCSTATUSTYPE& controlDtcStatusType);

    // record dtc
    void reportDTCEvent(uint32_t faultKey, uint8_t faultStatus);

    // listener session change for 85
    void reportSessionChange(uint32_t sessionType);

    // output dtc detail information
    void requestOutputDtcInfo();

    // output result
    void replyOutputDtcInfo(const bool result);

private:
    static std::mutex m_lck;
    static DiagServerEventHandler* m_instance;
    std::unique_ptr<ThreadPool> m_upThreadPool;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
