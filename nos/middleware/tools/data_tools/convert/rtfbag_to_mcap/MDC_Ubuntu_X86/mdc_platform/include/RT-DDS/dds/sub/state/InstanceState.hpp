/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATUS_INSTANCESTATE_HPP
#define SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATUS_INSTANCESTATE_HPP

#include <bitset>

namespace dds {
namespace sub {
namespace state {
class InstanceState {
public:
    /** we preset the bit width to 16 */
    using MaskType = std::bitset<16U>;

    static InstanceState Alive() noexcept;

    static InstanceState NotAliveDisposed() noexcept;

    static InstanceState NotAliveNoWriters() noexcept;

    static InstanceState Any() noexcept;

    static InstanceState NotAlive() noexcept;

    bool operator==(const InstanceState& rhs) const noexcept;

    bool operator!=(const InstanceState& rhs) const noexcept;
private:
    MaskType mask_{};
};
}
}
}


#endif // SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATUS_INSTANCESTATE_HPP
