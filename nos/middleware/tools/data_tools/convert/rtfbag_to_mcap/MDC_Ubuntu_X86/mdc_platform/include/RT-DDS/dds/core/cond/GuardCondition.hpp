/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: GuardCondition.hpp
 */

#ifndef DDS_CORE_COND_GUARD_CONDITION_HPP
#define DDS_CORE_COND_GUARD_CONDITION_HPP

#include <RT-DDS/dds/core/cond/Condition.hpp>

namespace dds {
namespace core {
namespace cond {
class GuardConditionImpl;
/**
 * @brief A condition whose trigger value is
 * under the control of the application.
 */
class GuardCondition : public Condition {
public:
    /**
     * @ingroup GuardCondition
     * @brief Constructor of GuardCondition.
     * @req{AR-iAOS-RCS-DDS-01401,
     * GuardCondition shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00090
     * }
     */
    GuardCondition() noexcept;

    /**
     * @ingroup GuardCondition
     * @brief Destructor of GuardCondition.
     * @req{AR-iAOS-RCS-DDS-01402,
     * GuardCondition shall support destruction process.,
     * QM,
     * DR-iAOS-RCS-DDS-00090
     * }
     */
    ~GuardCondition(void) override = default;
    /**
     * @ingroup GuardCondition
     * @brief Set trigger value of GuardCondition.
     * @param[in] value Set trigger value to this value.
     * @req{AR-iAOS-RCS-DDS-01403,
     * GuardCondition shall support setting TriggerValue.,
     * QM,
     * DR-iAOS-RCS-DDS-00090
     * }
     */
    void SetTriggerValue(bool value) const;
private:
    std::shared_ptr<GuardConditionImpl> impl_;
};
}
}
}

#endif /* DDS_CORE_COND_GUARD_CONDITION_HPP */

