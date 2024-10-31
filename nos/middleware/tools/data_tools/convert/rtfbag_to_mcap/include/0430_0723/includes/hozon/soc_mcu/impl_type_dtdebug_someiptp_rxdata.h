/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SOMEIPTP_RXDATA_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SOMEIPTP_RXDATA_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_uint16.h"
#include "impl_type_uint32.h"

namespace hozon {
namespace soc_mcu {
struct DtDebug_SomeIpTp_RxData {
    ::UInt8 InitState;
    ::UInt16 PduLength;
    ::UInt16 ReceivePayload;
    ::UInt16 ReceiveSumPayload;
    ::UInt16 bufferSize;
    ::UInt16 TP_Flag;
    ::UInt8 RequestId;
    ::UInt16 ProtocolVersion;
    ::UInt8 InterfaceVersion;
    ::UInt8 MessageType;
    ::UInt8 ReturnCode;
    ::UInt32 Offsetall;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(InitState);
        fun(PduLength);
        fun(ReceivePayload);
        fun(ReceiveSumPayload);
        fun(bufferSize);
        fun(TP_Flag);
        fun(RequestId);
        fun(ProtocolVersion);
        fun(InterfaceVersion);
        fun(MessageType);
        fun(ReturnCode);
        fun(Offsetall);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(InitState);
        fun(PduLength);
        fun(ReceivePayload);
        fun(ReceiveSumPayload);
        fun(bufferSize);
        fun(TP_Flag);
        fun(RequestId);
        fun(ProtocolVersion);
        fun(InterfaceVersion);
        fun(MessageType);
        fun(ReturnCode);
        fun(Offsetall);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("InitState", InitState);
        fun("PduLength", PduLength);
        fun("ReceivePayload", ReceivePayload);
        fun("ReceiveSumPayload", ReceiveSumPayload);
        fun("bufferSize", bufferSize);
        fun("TP_Flag", TP_Flag);
        fun("RequestId", RequestId);
        fun("ProtocolVersion", ProtocolVersion);
        fun("InterfaceVersion", InterfaceVersion);
        fun("MessageType", MessageType);
        fun("ReturnCode", ReturnCode);
        fun("Offsetall", Offsetall);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("InitState", InitState);
        fun("PduLength", PduLength);
        fun("ReceivePayload", ReceivePayload);
        fun("ReceiveSumPayload", ReceiveSumPayload);
        fun("bufferSize", bufferSize);
        fun("TP_Flag", TP_Flag);
        fun("RequestId", RequestId);
        fun("ProtocolVersion", ProtocolVersion);
        fun("InterfaceVersion", InterfaceVersion);
        fun("MessageType", MessageType);
        fun("ReturnCode", ReturnCode);
        fun("Offsetall", Offsetall);
    }

    bool operator==(const ::hozon::soc_mcu::DtDebug_SomeIpTp_RxData& t) const
    {
        return (InitState == t.InitState) && (PduLength == t.PduLength) && (ReceivePayload == t.ReceivePayload) && (ReceiveSumPayload == t.ReceiveSumPayload) && (bufferSize == t.bufferSize) && (TP_Flag == t.TP_Flag) && (RequestId == t.RequestId) && (ProtocolVersion == t.ProtocolVersion) && (InterfaceVersion == t.InterfaceVersion) && (MessageType == t.MessageType) && (ReturnCode == t.ReturnCode) && (Offsetall == t.Offsetall);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTDEBUG_SOMEIPTP_RXDATA_H
