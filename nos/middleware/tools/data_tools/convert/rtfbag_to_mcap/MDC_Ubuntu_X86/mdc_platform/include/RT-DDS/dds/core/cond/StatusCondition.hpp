/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: StatusCondition.hpp
 */

#ifndef DDS_CORE_COND_STATUS_CONDITION_HPP
#define DDS_CORE_COND_STATUS_CONDITION_HPP

#include <RT-DDS/dds/core/Entity.hpp>
#include <RT-DDS/dds/core/cond/Condition.hpp>
#include <RT-DDS/dds/core/status/StatusMask.hpp>

namespace dds {
namespace core {
namespace cond {
class StatusConditionImpl;

/**
 * @brief A condition associated with each dds::core::Entity
 */
class StatusCondition : public Condition {
public:
    /**
     * @brief Obtains a reference to the StatusCondition in an entity.
     * @param[in] entity The Entity whose status condition we're getting a reference
     * to. There is exactly one StatusCondition per Entity and one Entity per
     * StatusCondition.
     */
    explicit StatusCondition(const dds::core::Entity &entity) noexcept;

    ~StatusCondition(void) override = default;

    /**
     * @brief Defines the list of communication statuses that determine the
     * trigger value.
     * @param[in] status The list of communication statuses.
     */
    void EnabledStatuses(const dds::core::status::StatusMask &status);

    /**
     * @brief Gets the list of enabled statuses.
     * @return The list of enabled statuses.
     */
    dds::core::status::StatusMask EnabledStatuses() const;

private:
    std::shared_ptr<StatusConditionImpl> impl_;
};
}
}
}

#endif /* DDS_CORE_COND_STATUS_CONDITION_HPP */

