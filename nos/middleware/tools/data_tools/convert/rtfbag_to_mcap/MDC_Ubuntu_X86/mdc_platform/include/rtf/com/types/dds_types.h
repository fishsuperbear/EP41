/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: This provide the config to support that adapt to ros platform.
 * Create: 2020-04-21
 */
#ifndef RTF_COM_TYPES_DDS_TYPES_H
#define RTF_COM_TYPES_DDS_TYPES_H

#include "vrtf/driver/dds/dds_driver_types.h"
#include "rtf/com/types/error_code.h"
#include "rtf/rtfcm/parse_json_to_get_config.h"

namespace rtf {
namespace com {
namespace dds {
using DomainId      = int16_t;
using FragSize      = uint32_t;
using ListSize      = uint32_t;
using CacheSize     = int32_t;
using SdInfo        = vrtf::driver::dds::DDSServiceDiscoveryInfo;
using EventInfo     = vrtf::driver::dds::DDSEventInfo;
using MethodInfo    = vrtf::driver::dds::DDSMethodInfo;
using TransportQos  = vrtf::driver::dds::qos::TransportQos;
using DurabilityQos = vrtf::driver::dds::qos::DurabilityQos;
using ScheduleMode = vrtf::driver::dds::qos::ScheduleMode;
using ReliabilityKind = vrtf::driver::dds::qos::ReliabilityKind;
using DiscoveryFilter = vrtf::driver::dds::qos::DiscoveryFilter;
using ParticipantQos = vrtf::driver::dds::qos::ParticipantQos;
using DiscoveryConfigQos = vrtf::driver::dds::qos::DiscoveryConfigQos;
using TransportMode = vrtf::driver::dds::qos::TransportMode;

struct BandWidthInfo {
    uint32_t bandWidth;
    uint32_t sendWindow;
};

class QosBase {
public:
    QosBase() : historyDepth_(100) // Default history depth: 100
    {
        using namespace ara::godel::common::log;
        Log::InitLog("", "CM", "CM_LOG");
        logInstance_ = Log::GetLog("CM");
    }
    virtual ~QosBase() = default;
    virtual rtf::com::ErrorCode SetHistoryDepth(const int32_t& size) noexcept
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&size);
#endif
        rtf::com::ErrorCode ret;
        if (size >= 1 && size <= 1000) { // The range of qos history depth is [1, 1000]
            ret = rtf::com::ErrorCode::OK;
        } else {
            ret = rtf::com::ErrorCode::PARAMETER_ERROR;
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->warn() << "History depth value must be in [1, 1000].";
        }
        historyDepth_ = size;
        return ret;
    }
    virtual int32_t GetHistoryDepth() const noexcept
    {
        return historyDepth_;
    }

protected:
    QosBase(const QosBase& other) = default;
    QosBase& operator=(const QosBase& other) = default;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
private:
    int32_t historyDepth_;
};

class WriterQos : public QosBase {
public:
    WriterQos()
    {
        transportMode_ = rtf::rtfcm::rtfmaintaind::ParseJsonToGetConfig::GetInstance()->GetTransportMode();
    }
    ~WriterQos() override = default;
    void SetBandWidth(BandWidthInfo const &info) noexcept
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&info);
#endif
        isBandWidthSet_ = true;
        bandWidthInfo_ = info;
    }
    BandWidthInfo GetBandWidth() const noexcept
    {
        return bandWidthInfo_;
    }
    bool IsBandWidthSet() const noexcept
    {
        return isBandWidthSet_;
    }

    void SetTransportMode(const TransportMode &mode) noexcept
    {
        if (mode > dds::TransportMode::TRANSPORT_SYNCHRONOUS_MODE) {
            logInstance_->warn() << "[Invalid transport mode][mode=" << static_cast<uint32_t>(mode) << "]";
        } else {
            transportMode_ = mode;
        }
    }

    TransportMode GetTransportMode() const noexcept
    {
        return transportMode_;
    }
private:
    // bandWidth Mbit/s, [0,1000], default: 0; sendWindow ms, [1, 1000], default: 8
    BandWidthInfo bandWidthInfo_ {0, 8};
    bool isBandWidthSet_ = false;
    TransportMode transportMode_{TransportMode::TRANSPORT_ASYNCHRONOUS_MODE};
};

class ReaderQos : public QosBase {
public:
    ReaderQos() : enableMulticastAddr_(false){}
    ~ReaderQos() override = default;
    void SetMulticastAddr(const std::string& addr = vrtf::driver::dds::qos::DEFAULT_MULTICAST_ADDR) noexcept
    {
/* Axivion Next Line AutosarC++19_03-A16.0.1 : AOS_TAINT used for stigma modeling */
#ifdef AOS_TAINT
        Coverity_Tainted_Set((void *)&addr);
#endif
        enableMulticastAddr_ = true;
        multicastAddr_ = addr;
    }
    std::string GetMulticastAddr() const noexcept
    {
        return multicastAddr_;
    }
    bool IsMulticastEnabled() const noexcept
    {
        return enableMulticastAddr_;
    }
private:
    std::string multicastAddr_;
    bool enableMulticastAddr_;
};
} // namespace dds
} // namespace com
} // namespace rtf

#endif // RTF_COM_TYPES_DDS_TYPES_H
