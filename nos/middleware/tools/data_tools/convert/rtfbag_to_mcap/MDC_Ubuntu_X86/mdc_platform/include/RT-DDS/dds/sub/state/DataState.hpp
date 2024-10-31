/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:
 */

#ifndef SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATE_DATASTATE_HPP
#define SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATE_DATASTATE_HPP

#include <RT-DDS/dds/sub/state/InstanceState.hpp>
#include <RT-DDS/dds/sub/state/ViewState.hpp>

namespace dds {
namespace sub {
namespace state {
class DataState {
public:
    const InstanceState& GetInstanceState() const noexcept
    {
        return instanceState_;
    }

    const ViewState& GetViewState() const noexcept
    {
        return viewState_;
    }

    void SetInstanceState(const InstanceState& instanceState) noexcept
    {
        instanceState_ = instanceState;
    }

    void SetViewState(const ViewState& viewState) noexcept
    {
        viewState_ = viewState;
    }

private:
    InstanceState instanceState_;
    ViewState viewState_;
};
}
}
}


#endif // SRC_DCPS_API_ISOCPP_SRC_DDS_SUB_STATE_DATASTATE_HPP
