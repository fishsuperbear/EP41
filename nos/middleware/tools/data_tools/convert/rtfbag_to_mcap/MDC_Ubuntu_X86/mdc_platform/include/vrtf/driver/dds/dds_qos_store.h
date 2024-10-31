/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: use to store dds qos config
 * Create: 2019-12-17
 */
#ifndef VRTF_VCC_DRIVER_DDS_DDSQOSSTORE_H
#define VRTF_VCC_DRIVER_DDS_DDSQOSSTORE_H
#include <set>
#include "vrtf/vcc/api/types.h"
namespace vrtf {
namespace driver {
namespace dds {
namespace qos {
const std::string DEFAULT_MULTICAST_ADDR = "defaultMulticastAddr";
class DiscoveryFilter {
public:
    DiscoveryFilter(const uint8_t& id, const std::string& idFilter)
        : classificationId_(id), classificationIdFilter_(idFilter) {}
    virtual ~DiscoveryFilter() = default;
    uint8_t GetClassificationId() const noexcept
    {
        return classificationId_;
    }
    std::string GetClassificationIdFilter() const noexcept
    {
        return classificationIdFilter_;
    }
private:
    uint8_t       classificationId_;
    std::string   classificationIdFilter_;
};

struct DiscoveryConfigQos {
    uint8_t announcements = 1;
    uint32_t minAnnouncementInterval = 500;
    uint32_t maxAnnouncementInterval = 1000;
    uint32_t leaseDuration = 60000;
    uint32_t assertPeriod = 24000;
};

class ParticipantQos {
public:
    ParticipantQos() = default;
    explicit ParticipantQos(DiscoveryFilter const &filter) : discoveryFilter_(filter) {}
    virtual ~ParticipantQos() = default;
    // Just for built-in app temporarily, like PHM/TOOLS
    DiscoveryFilter GetDiscoveryFilter() const noexcept
    {
        return discoveryFilter_;
    }
    void SetDiscoveryConfigQos(DiscoveryConfigQos const &discoveryConfigQos) noexcept
    {
        discoveryConfigQos_ = discoveryConfigQos;
    }
    DiscoveryConfigQos GetDiscoveryConfigQos() const noexcept
    {
        return discoveryConfigQos_;
    }
private:
    // Just for built-in app temporarily, like PHM/TOOLS
    DiscoveryFilter discoveryFilter_ = DiscoveryFilter(0, "UNDEFINED_DISCOVERY_FILTER");
    DiscoveryConfigQos discoveryConfigQos_ {1, 500, 1000, 60000, 24000};
};

struct BandWidthInfo {
    uint32_t bandWidth;
    uint32_t sendWindow;
};

enum class DurabilityQos : std::uint8_t {
    VOLATILE = 0x00,
    TRANSIENT_LOCAL
};

enum class HistoryQos : std::uint8_t {
    KEEP_LAST = 0x00,
    KEEP_ALL
};

enum class ReliabilityKind : std::uint8_t {
    BEST_EFFORT = 0x00,
    RELIABLE
};

enum class TransportQos : std::uint8_t {
    UDP = 0x00,
    ICC,
    SHM,
    DSHM,
    UDP2DSHM
};
enum class SHMPER : std::uint8_t {
    SHM_RWRWOO_MODE = 0x00,
    SHM_RWRWRW_MODE,
};

enum class ScheduleMode : uint8_t {
    DETERMINATE   = 0x00U,
    INDETERMINATE = 0x01U
};

enum class TransportMode : std::uint8_t {
    TRANSPORT_ASYNCHRONOUS_MODE,
    TRANSPORT_SYNCHRONOUS_MODE
};
class DDSEventQosStore {
public:
    DDSEventQosStore() = default;
    ~DDSEventQosStore() = default;
    inline void SetDurabilityQos(const DurabilityQos& qos)
    {
        durabilityQos_ = qos;
    }
    inline void SetParticipantTransportQos(const std::set<TransportQos> transportMode)
    {
        transportQos_ = transportMode;
    }
    inline DurabilityQos GetDurabilityQos() const
    {
        return durabilityQos_;
    }
    inline void SetReliabilityKind(const ReliabilityKind& kind)
    {
        reliabilityKind_ = kind;
    }
    inline ReliabilityKind GetReliabilityKind() const
    {
        return reliabilityKind_;
    }
    inline std::set<TransportQos> GetParticipantTransportQos() const
    {
        return transportQos_;
    }
    inline void SetFragsize(const uint32_t& fragsize)
    {
        fragSize_ = fragsize;
    }
    inline void SetListsize(const uint32_t& listsize)
    {
        listSize_ = listsize;
    }
    inline std::uint32_t GetFragsize() const
    {
        return fragSize_;
    }
    inline std::uint32_t GetListsize() const
    {
        return listSize_;
    }

    void SetHistoryQos(const qos::HistoryQos historyQos)
    {
        historyQos_ = historyQos;
    }

    qos::HistoryQos GetHistoryQos() const
    {
        return historyQos_;
    }

    /**
     * @brief Support to ignore SHM check in writer, when writer fragsize and listsize settings are different from
     *  already existed reader, and reader's fragsize and listsize setting will be used.
     * @param[in] true, ignore SHM check; false(default), do SHM check.
     */
    void IgnoreSHMCheck(const bool& isCheck)
    {
        isSHMCheckIgnored_ = isCheck;
    }
    /**
     * @brief Check if SHM check is ignored.
     * @return true, ignore SHM check; false(default), do SHM check
     */
    bool IsSHMCheckIgnored() const noexcept
    {
        return isSHMCheckIgnored_;
    }

    std::string GetMulticastAddr() const noexcept
    {
        return multicastAddr_;
    }

    void SetMulticastAddr(const std::string& addr) noexcept
    {
        enableMulticastAddr_ = true;
        multicastAddr_ = addr;
    }

    bool IsMulticastEnabled() const noexcept
    {
        return enableMulticastAddr_;
    }
    void SetBandWidthInfo(BandWidthInfo const &info) noexcept
    {
        isBandWidthSet_ = true;
        bwInfo_ = info;
    }
    BandWidthInfo GetBandWidthInfo() const noexcept
    {
        return bwInfo_;
    }
    bool IsBandWidthSet() const noexcept
    {
        return isBandWidthSet_;
    }

    void SetTransportMode(qos::TransportMode mode) noexcept
    {
        transportMode_ = mode;
    }

    qos::TransportMode GetTransportMode() const noexcept
    {
        return transportMode_;
    }
private:
    DurabilityQos durabilityQos_ = DurabilityQos::VOLATILE;
    ReliabilityKind reliabilityKind_ = ReliabilityKind::RELIABLE;
    std::set<TransportQos> transportQos_ = {TransportQos::UDP};
    std::uint32_t fragSize_ = 2048;  // default fragSize value
    std::uint32_t listSize_ = 256;  // default listSize value
    std::string multicastAddr_;
    bool enableMulticastAddr_ = false;
    HistoryQos historyQos_ = HistoryQos::KEEP_LAST;
    bool isSHMCheckIgnored_ = false;
    bool isBandWidthSet_ = false;
    BandWidthInfo bwInfo_ {0, 8};
    qos::TransportMode transportMode_{qos::TransportMode::TRANSPORT_ASYNCHRONOUS_MODE};
};
class DDSMethodQosStore {
public:
    DDSMethodQosStore(){}
    ~DDSMethodQosStore(void) = default;
    inline void SetRequestDurabilityQos(const DurabilityQos& qos)
    {
        requestDurabilityQos_ = qos;
    }
    inline void SetReplyDurabilityQos(const DurabilityQos& qos)
    {
        replyDurabilityQos_ = qos;
    }
    inline void SetReliabilityKind(const ReliabilityKind& kind)
    {
        reliabilityKind_ = kind;
    }
    inline ReliabilityKind GetReliabilityKind() const noexcept
    {
        return reliabilityKind_;
    }
    inline void SetParticipantTransport(const std::set<TransportQos> transportMode)
    {
        transportQos_ = transportMode;
    }
    inline DurabilityQos GetRequestDurabilityQos() const
    {
        return requestDurabilityQos_;
    }
    inline DurabilityQos GetReplyDurabilityQos() const
    {
        return replyDurabilityQos_;
    }
    inline std::set<TransportQos> GetParticipantTransport() const
    {
        return transportQos_;
    }
    inline void SetRequestFragSize(const std::uint32_t fragsize)
    {
        requestFragSize_ = fragsize;
    }
    inline void SetRequestListSize(const std::uint32_t listsize)
    {
        requestListSize_ = listsize;
    }
    inline void SetReplyFragSize(const std::uint32_t fragsize)
    {
        replyFragSize_ = fragsize;
    }
    inline void SetReplyListSize(const std::uint32_t listsize)
    {
        replyListSize_ = listsize;
    }
    inline std::uint32_t GetRequestFragSize() const
    {
        return requestFragSize_;
    }
    inline std::uint32_t GetReplyFragSize() const
    {
        return replyFragSize_;
    }
    inline std::uint32_t GetRequestListSize() const
    {
        return requestListSize_;
    }
    inline std::uint32_t GetReplyListSize() const
    {
        return replyListSize_;
    }
    inline void SetRequestHistoryQos(const qos::HistoryQos historyQos)
    {
        requestHistoryQos_ = historyQos;
    }
    inline qos::HistoryQos GetRequestHistoryQos() const
    {
        return requestHistoryQos_;
    }
    inline void SetReplyHistoryQos(const qos::HistoryQos historyQos)
    {
        replyHistoryQos_ = historyQos;
    }
    inline qos::HistoryQos GetReplyHistoryQos() const
    {
        return replyHistoryQos_;
    }
    void SetRequestSharedMemoryAuthority(const SHMPER& shmFileMode)
    {
        reuestShmPer_ = shmFileMode;
    }
    SHMPER GetRequestSharedMemoryAuthority() const
    {
        return reuestShmPer_;
    }
    void SetReplySharedMemoryAuthority(const SHMPER& shmFileMode)
    {
        replyShmPer_ = shmFileMode;
    }
    SHMPER GetReplySharedMemoryAuthority() const
    {
        return replyShmPer_;
    }
    void SetRequestWriterHistoryDepth(const std::int32_t& writerHistoryDepth)
    {
        requestWriterHistoryDepth_ = writerHistoryDepth;
    }
    std::int32_t GetRequestWriterHistoryDepth() const
    {
        return requestWriterHistoryDepth_;
    }
    void SetReplyWriterHistoryDepth(const std::int32_t& writerHistoryDepth)
    {
        replyWriterHistoryDepth_ = writerHistoryDepth;
    }
    std::int32_t GetReplyWriterHistoryDepth() const
    {
        return replyWriterHistoryDepth_;
    }
    void SetRequestReaderHistoryDepth(const std::int32_t& readerHistoryDepth)
    {
        requestReaderHistoryDepth_ = readerHistoryDepth;
    }
    std::int32_t GetRequestReaderHistoryDepth() const
    {
        return requestReaderHistoryDepth_;
    }
    void SetReplyReaderHistoryDepth(const std::int32_t& readerHistoryDepth)
    {
        replyReaderHistoryDepth_ = readerHistoryDepth;
    }
    std::int32_t GetReplyReaderHistoryDepth() const
    {
        return replyReaderHistoryDepth_;
    }

    BandWidthInfo GetBandWidthInfo() const noexcept
    {
        return bwInfo_;
    }

    void SetBandWidthInfo(BandWidthInfo const &info) noexcept
    {
        isBandWidthSet_ = true;
        bwInfo_ = info;
    }

    bool IsBandWidthSet() const noexcept
    {
        return isBandWidthSet_;
    }

    void SetRequestWriterTransportMode(qos::TransportMode mode) noexcept
    {
        requestWriterTransportMode_ = mode;
    }

    qos::TransportMode GetRequestWriterTransportMode() const noexcept
    {
        return requestWriterTransportMode_;
    }

    void SetReplyWriterTransportMode(qos::TransportMode mode) noexcept
    {
        replyWriterTransportMode_ = mode;
    }

    qos::TransportMode GetReplyWriterTransportMode() const noexcept
    {
        return replyWriterTransportMode_;
    }
private:
    std::set<TransportQos> transportQos_ = {TransportQos::UDP};
    DurabilityQos requestDurabilityQos_ = DurabilityQos::VOLATILE;
    DurabilityQos replyDurabilityQos_ = DurabilityQos::VOLATILE;
    ReliabilityKind reliabilityKind_ = ReliabilityKind::RELIABLE;
    HistoryQos requestHistoryQos_ = HistoryQos::KEEP_LAST;
    HistoryQos replyHistoryQos_ = HistoryQos::KEEP_LAST;
    SHMPER reuestShmPer_ = SHMPER::SHM_RWRWOO_MODE; // dds shm file default authority is 660.
    SHMPER replyShmPer_ = SHMPER::SHM_RWRWOO_MODE; // dds shm file default authority is 660.
    uint32_t requestFragSize_ = 2048; // dds default fragsize number
    uint32_t requestListSize_ = 256; // dds default fragsize number
    uint32_t replyFragSize_ = 2048; // dds default fragsize number
    uint32_t replyListSize_ = 256;
    std::int32_t requestWriterHistoryDepth_ = 100; // 100 is the defaule value of request writer depth
    std::int32_t requestReaderHistoryDepth_ = 100; // 100 is the defaule value of request reader depth
    std::int32_t replyWriterHistoryDepth_ = 100; // 100 is the defaule value of reply writer depth
    std::int32_t replyReaderHistoryDepth_ = 100; // 100 is the defaule value of reply reader depth
    bool isBandWidthSet_ = false;
    BandWidthInfo bwInfo_ {0, 8};
    qos::TransportMode requestWriterTransportMode_{qos::TransportMode::TRANSPORT_ASYNCHRONOUS_MODE};
    qos::TransportMode replyWriterTransportMode_{qos::TransportMode::TRANSPORT_ASYNCHRONOUS_MODE};
};
}
}
}
}
#endif
