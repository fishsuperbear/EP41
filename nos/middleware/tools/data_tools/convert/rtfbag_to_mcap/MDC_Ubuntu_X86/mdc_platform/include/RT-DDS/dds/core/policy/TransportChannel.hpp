/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: TransportChannel.hpp
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_CHANNEL_HPP
#define DDS_CORE_POLICY_TRANSPORT_CHANNEL_HPP

#include <RT-DDS/dds/core/policy/TransportChannelKind.hpp>
#include <RT-DDS/dds/core/policy/TransportChannelShmPermission.hpp>
#include <RT-DDS/dds/core/Types.hpp>

namespace dds {
namespace core {
namespace policy {
class TransportChannel {
public:
    TransportChannel(void) = default;

    TransportChannel(
        TransportChannelKind kind, uint32_t shmId,
        uint32_t fragSize, uint32_t listSize, bool weakShmPara, bool determinate, bool enableUDPMulticast,
        std::string udpMulticastAddress, bool useMemPool = false) noexcept
        : kind_(kind), shmId_(shmId), fragSize_(fragSize), listSize_(listSize),
          weakShmPara_(weakShmPara), determinate_(determinate), enableUdpMulticast_(enableUDPMulticast),
          udpMulticastAddress_(std::move(udpMulticastAddress)), useMemPool_(useMemPool)
    {}

    ~TransportChannel(void) = default;

    void Kind(TransportChannelKind kind) noexcept
    {
        kind_ = kind;
    }

    void ShmId(uint32_t shmId) noexcept
    {
        shmId_ = shmId;
    }

    void FragSize(uint32_t fragSize) noexcept
    {
        fragSize_ = fragSize;
    }

    void ListSize(uint32_t listSize) noexcept
    {
        listSize_ = listSize;
    }

    /**
     * @brief This parameter is used to enable the memory pool function in the DSHM channel.
     * It does not take effect on the BSHM.
     * @param useFlag[in] flag of using memPool
     */
    void UseMemPool(bool useMemPoolFlag) noexcept
    {
        useMemPool_ = useMemPoolFlag;
    }

    void ShmPermission(TransportChannelShmPermission p)  noexcept
    {
        permission_ = p;
    }

    void WeakShmPara(bool weakFlag) noexcept
    {
        weakShmPara_ = weakFlag;
    }

    void Determinate(bool d) noexcept
    {
        determinate_ = d;
    }

    void EnableUDPMulticast(bool e) noexcept
    {
        enableUdpMulticast_ = e;
    }

    void UDPMulticastAddress(std::string s) noexcept
    {
        udpMulticastAddress_ = std::move(s);
    }

    TransportChannelKind Kind(void) const noexcept
    {
        return kind_;
    }

    uint32_t ShmId(void) const noexcept
    {
        return shmId_;
    }

    /**
     * @brief This parameter is used to enable the memory pool function in the DSHM channel.
     * It does not take effect on the BSHM.
     * @param useFlag[in] flag of using memPool
     */
    bool UseMemPool(void) const noexcept
    {
        return useMemPool_;
    }

    uint32_t FragSize(void) const noexcept
    {
        return fragSize_;
    }

    uint32_t ListSize(void) const noexcept
    {
        return listSize_;
    }

    TransportChannelShmPermission ShmPermission(void) const noexcept
    {
        return permission_;
    }

    bool Determinate(void) const noexcept
    {
        return determinate_;
    }

    bool WeakShmPara(void) const noexcept
    {
        return weakShmPara_;
    }

    bool EnableUDPMulticast(void) const noexcept
    {
        return enableUdpMulticast_;
    }

    const std::string& UDPMulticastAddress(void) const noexcept
    {
        return udpMulticastAddress_;
    }
private:
    TransportChannelKind kind_{TransportChannelKind::UDP};
    uint32_t shmId_{0U};
    uint32_t fragSize_{0U};
    uint32_t listSize_{0U};
    TransportChannelShmPermission permission_{TransportChannelShmPermission::RWRWOO_MODE};
    bool weakShmPara_{false};
    bool determinate_{true};
    bool enableUdpMulticast_{false};
    std::string udpMulticastAddress_{};
    bool useMemPool_{false};
};
}
}
}

#endif /* DDS_CORE_POLICY_TRANSPORT_CHANNEL_HPP */

