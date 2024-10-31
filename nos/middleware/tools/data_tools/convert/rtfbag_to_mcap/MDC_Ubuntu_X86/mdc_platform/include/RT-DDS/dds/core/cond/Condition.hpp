/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Condition.hpp
 */

#ifndef DDS_CORE_CONDITION_HPP
#define DDS_CORE_CONDITION_HPP

#include <vector>

#include <RT-DDS/dds/core/Reference.hpp>

namespace dds {
namespace core {
namespace cond {
class ConditionImpl;

/**
 * @brief Abstract base class of all the conditions
 */
class Condition : public dds::core::Reference<ConditionImpl> {
public:
    using ConditionImplPtr = std::shared_ptr<ConditionImpl>;

    /**
     * @brief Default constructor
     * @param[in] impl Smart pointer of the existing object.
     */
    explicit Condition(ConditionImplPtr impl) noexcept : Reference(std::move(impl))
    {}

    ~Condition() override = default;

    /**
     * @ingroup Condition
     * @brief Get the value of trigger value of the condition.
     * @req{AR-iAOS-RCS-DDS-01501,
     * Condition shall support getting TriggerValue.,
     * QM,
     * DR-iAOS-RCS-DDS-00090, DR-iAOS-RCS-DDS-00002, DR-iAOS-RCS-DDS-00091
     * }
     * @return bool The value of trigger value.
     */
    bool TriggerValue() const;

protected:
    Condition() noexcept : Reference(nullptr)
    {}

    friend class WaitSet;
};

using ConditionSeq = std::vector<dds::core::cond::Condition>;
}
}
}

#endif /* DDS_CORE_CONDITION_HPP */

