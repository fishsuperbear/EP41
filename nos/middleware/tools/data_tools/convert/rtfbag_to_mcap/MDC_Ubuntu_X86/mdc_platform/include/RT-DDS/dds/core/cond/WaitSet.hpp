/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: WaitSet.hpp
 */

#ifndef DDS_CORE_COND_WAIT_SET_HPP
#define DDS_CORE_COND_WAIT_SET_HPP

#include <RT-DDS/dds/core/Reference.hpp>
#include <RT-DDS/dds/core/ReturnCode.hpp>
#include <RT-DDS/dds/core/Duration.hpp>
#include <RT-DDS/dds/core/cond/Condition.hpp>

namespace dds {
namespace core {
namespace cond {
class WaitSetImpl;

/**
 * @brief Allows an application to wait until one or more of the attached Condition
 * objects have a trigger_value of true or else until the timeout expires.
 */
class WaitSet : public dds::core::Reference<WaitSetImpl> {
public:
    /**
     * @ingroup WaitSet
     * @brief Constructor of waitSet.
     * @req{AR-iAOS-RCS-DDS-01101,
     * WaitSet shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00088
     * }
     */
    WaitSet(void) noexcept;

    ~WaitSet(void) override = default;

    /**
     * @ingroup WaitSet
     * @brief Attaches a Condition to the WaitSet.
     * @req{AR-iAOS-RCS-DDS-01103,
     * WaitSet shall support attaching a Condition.,
     * QM,
     * DR-iAOS-RCS-DDS-00089
     * }
     * @param[in] cond Condition to be attached.
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     */
    dds::core::ReturnCode AttachCondition(const dds::core::cond::Condition &cond);

    /**
     * @ingroup WaitSet
     * @brief Detach a Condition from the WaitSet.
     * req{AR-iAOS-RCS-DDS-01104,
     * WaitSet shall support detaching a Condition.,
     * QM,
     * DR-iAOS-RCS-DDS-00089
     * }
     * @param[in] cond Condition to be attached.
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     */
    dds::core::ReturnCode DetachCondition(const dds::core::cond::Condition &cond);

    /**
     * @brief Retrieves the list of attached conditions.
     * @return The list of attached conditions.
     */
    ConditionSeq GetConditions(void) const;

    /**
     * @ingroup WaitSet
     * @brief Allows an application thread to wait for the occurrence of certain conditions.
     * req{AR-iAOS-RCS-DDS-01105,
     * WaitSet shall support waiting until attached Conditions triggered.,
     * QM,
     * DR-iAOS-RCS-DDS-00088
     * }
     * @param[in] timeout a wait timeout
     * @return A vector containing the active conditions or an empty vector if
     *         the operation times out.
     */
    ConditionSeq Wait(dds::core::Duration timeout) const;
};
}
}
}

#endif /* DDS_CORE_COND_WAIT_SET_HPP */

