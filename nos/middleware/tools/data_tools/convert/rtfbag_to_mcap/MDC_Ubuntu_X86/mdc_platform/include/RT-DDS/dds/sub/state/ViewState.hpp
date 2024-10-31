/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATE_VIEWSTATE_HPP
#define SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATE_VIEWSTATE_HPP

#include <bitset>

namespace dds {
namespace sub {
namespace state {
class ViewState {
public:
    /** we preset the bit width to 16 */
    using MaskType = std::bitset<16U>;

    static ViewState New() noexcept;

    static ViewState NotNew() noexcept;

    static ViewState Any() noexcept;

    bool operator==(const ViewState& rhs) const noexcept;

    bool operator!=(const ViewState& rhs) const noexcept;

private:
    MaskType mask_{};
};
}
}
}

#endif // SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATE_VIEWSTATE_HPP
