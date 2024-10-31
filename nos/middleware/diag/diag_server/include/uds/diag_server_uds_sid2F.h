/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid2F.h is designed for diagnostic InputOutput Control.
 */

#ifndef DIAG_SERVER_UDS_SID2F_H
#define DIAG_SERVER_UDS_SID2F_H

#include <unordered_map>
#include <mutex>
#include "diag/diag_server/include/uds/diag_server_uds_base.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerUdsSid2F : public DiagServerUdsBase {
public:
    DiagServerUdsSid2F();
    virtual ~DiagServerUdsSid2F();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    void initMap();

private:
    DiagServerUdsSid2F(const DiagServerUdsSid2F &);
    DiagServerUdsSid2F & operator = (const DiagServerUdsSid2F &);

private:
    static std::mutex mtx_;
    std::unordered_map<uint8_t, DiagServerUdsSid2F*> sub_map_;
};

class DiagServerUdsSid2FReturnControlToECU  : public DiagServerUdsSid2F {
public:
    DiagServerUdsSid2FReturnControlToECU ();
    virtual ~DiagServerUdsSid2FReturnControlToECU ();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid2FReturnControlToECU (const DiagServerUdsSid2FReturnControlToECU  &);
    DiagServerUdsSid2FReturnControlToECU & operator = (const DiagServerUdsSid2FReturnControlToECU  &);

};

class DiagServerUdsSid2FResetToDefault : public DiagServerUdsSid2F {
public:
    DiagServerUdsSid2FResetToDefault();
    virtual ~DiagServerUdsSid2FResetToDefault();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);

private:
    DiagServerUdsSid2FResetToDefault(const DiagServerUdsSid2FResetToDefault &);
    DiagServerUdsSid2FResetToDefault & operator = (const DiagServerUdsSid2FResetToDefault &);

};

class DiagServerUdsSid2FFreezeCurrentStatus : public DiagServerUdsSid2F {
public:
    DiagServerUdsSid2FFreezeCurrentStatus();
    virtual ~DiagServerUdsSid2FFreezeCurrentStatus();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);
    uint32_t GetSeed(const uint8_t level);

private:
    DiagServerUdsSid2FFreezeCurrentStatus(const DiagServerUdsSid2FFreezeCurrentStatus &);
    DiagServerUdsSid2FFreezeCurrentStatus & operator = (const DiagServerUdsSid2FFreezeCurrentStatus &);

};

class DiagServerUdsSid2FShortTermAdjustment : public DiagServerUdsSid2F {
public:
    DiagServerUdsSid2FShortTermAdjustment();
    virtual ~DiagServerUdsSid2FShortTermAdjustment();
    virtual void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

protected:
    virtual void NegativeResponse(const DiagServerUdsMessage& udsMessage);
    uint8_t CompareKey(const uint8_t level, const uint32_t seed, const uint32_t key);

private:
    DiagServerUdsSid2FShortTermAdjustment(const DiagServerUdsSid2FShortTermAdjustment &);
    DiagServerUdsSid2FShortTermAdjustment & operator = (const DiagServerUdsSid2FShortTermAdjustment &);
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_UDS_SID2F_H
