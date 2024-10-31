/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_uds_sid2F.cpp is designed for diagnostic InputOutput Control.
 */

#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid2F.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerUdsSid2F::mtx_;

DiagServerUdsSid2F::DiagServerUdsSid2F()
{
}

DiagServerUdsSid2F::~DiagServerUdsSid2F()
{
}

void
DiagServerUdsSid2F::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    initMap();
    std::vector<uint8_t> sendData;
    sendData.push_back(DIAG_SERVER_SERVICE_REQUEST_OPC_SECURITY_ACCESS);

}

void
DiagServerUdsSid2F::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    std::vector<uint8_t> sendData;
    sendData.push_back(DIAG_SERVER_SERVICE_REQUEST_OPC_SECURITY_ACCESS);

}

void
DiagServerUdsSid2F::initMap()
{
    sub_map_.insert(std::make_pair(0x00, new DiagServerUdsSid2FReturnControlToECU()));
    sub_map_.insert(std::make_pair(0x01, new DiagServerUdsSid2FResetToDefault()));
    sub_map_.insert(std::make_pair(0x02, new DiagServerUdsSid2FFreezeCurrentStatus()));
    sub_map_.insert(std::make_pair(0x03, new DiagServerUdsSid2FShortTermAdjustment()));

}

DiagServerUdsSid2FReturnControlToECU::DiagServerUdsSid2FReturnControlToECU()
: DiagServerUdsSid2F()
{
}

DiagServerUdsSid2FReturnControlToECU::~DiagServerUdsSid2FReturnControlToECU()
{
}

void
DiagServerUdsSid2FReturnControlToECU::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{

}

void
DiagServerUdsSid2FReturnControlToECU::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{

}

DiagServerUdsSid2FResetToDefault::DiagServerUdsSid2FResetToDefault()
: DiagServerUdsSid2F()
{
}

DiagServerUdsSid2FResetToDefault::~DiagServerUdsSid2FResetToDefault()
{
}

void
DiagServerUdsSid2FResetToDefault::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{

}

void
DiagServerUdsSid2FResetToDefault::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{

}

DiagServerUdsSid2FFreezeCurrentStatus::DiagServerUdsSid2FFreezeCurrentStatus()
: DiagServerUdsSid2F()
{
}

DiagServerUdsSid2FFreezeCurrentStatus::~DiagServerUdsSid2FFreezeCurrentStatus()
{
}

void
DiagServerUdsSid2FFreezeCurrentStatus::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{

}

void
DiagServerUdsSid2FFreezeCurrentStatus::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{

}

DiagServerUdsSid2FShortTermAdjustment::DiagServerUdsSid2FShortTermAdjustment()
: DiagServerUdsSid2F()
{
}

DiagServerUdsSid2FShortTermAdjustment::~DiagServerUdsSid2FShortTermAdjustment()
{
}

void
DiagServerUdsSid2FShortTermAdjustment::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{

}

void
DiagServerUdsSid2FShortTermAdjustment::NegativeResponse(const DiagServerUdsMessage& udsMessage)
{

}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
