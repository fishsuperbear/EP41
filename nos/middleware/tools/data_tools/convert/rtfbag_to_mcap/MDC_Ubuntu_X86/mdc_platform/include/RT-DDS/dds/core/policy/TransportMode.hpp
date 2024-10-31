/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: TransportPriority.hpp
 */

#ifndef DDS_CORE_POLICY_TRANSPORT_MODE_HPP
#define DDS_CORE_POLICY_TRANSPORT_MODE_HPP

#include <RT-DDS/dds/core/policy/TransportModeKind.hpp>

namespace dds {
namespace core {
namespace policy {
/* @brief Set up synchronous/asynchronous sending via TransportMode Qos */
class TransportMode {
public:
    /**
      * @ingroup dds::core::policy
      * @brief TransportMode constructor
      * @param[in] NONE
      * @return constructor
      * @req{AR-iAOS-RCS-DDS-20028, AR20221205568478
      * DDS supports synchronous/asynchronous sending of data by setting TransportMode,
      * SR-iAOS3-RCS-DDS-00055
      * }
      */
    TransportMode() = default;

    /**
    * @brief The default setting is TransportMode to send synchronously
    * @param[in] TransportModeKind an enum to specify which transportMode to be set
    */
    explicit TransportMode(TransportModeKind kind) noexcept : transportModeTypeKind_(kind)
    {}
    /**
   * @ingroup dds::core::policy
   * @brief TransportMode Destructor
   * @param[in] NONE
   * @return Destructor
   * @req{AR-iAOS-RCS-DDS-20028, AR20221205568478
   * DDS supports synchronous/asynchronous sending of data by setting TransportMode,
   * SR-iAOS3-RCS-DDS-00055
   * }
   */
    ~TransportMode() = default;
    /**
   * @ingroup dds::core::policy
   * @brief TransportModeKind Setter
   * @param[in] TransportModeKind an enum to specify which transportMode to be set
   * @return void
   * @req{AR-iAOS-RCS-DDS-20028, AR20221205568478
   * DDS supports synchronous/asynchronous sending of data by setting TransportMode,
   * SR-iAOS3-RCS-DDS-00055
   * }
   */
    void SetTransportModeKind(const TransportModeKind &kind) noexcept
    {
        transportModeTypeKind_ = kind;
    }

    /**
    * @ingroup dds::core::policy
    * @brief TransportModeKind Getter
    * @param[in] TransportModeKind an enum to specify which TransportMode to be got
    * @return the value of correspoding TransportMode
    * @req{AR-iAOS-RCS-DDS-20028, AR20221205568478
    * DDS supports synchronous/asynchronous sending of data by setting TransportMode,
    * SR-iAOS3-RCS-DDS-00055
    * }
    */
    dds::core::policy::TransportModeKind GetTransportModeKind() const noexcept
    {
        return transportModeTypeKind_;
    }

private:
    TransportModeKind transportModeTypeKind_ {TransportModeKind::TRANSPORT_ASYNCHRONOUS_MODE};
};
}
}
}

#endif /* DDS_CORE_POLICY_TRANSPORT_MODE_HPP */

