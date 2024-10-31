/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Entity.hpp
 */

#ifndef DDS_CORE_ENTITY_HPP
#define DDS_CORE_ENTITY_HPP

#include <RT-DDS/dds/core/Reference.hpp>
#include <RT-DDS/dds/core/ReturnCode.hpp>
#include <RT-DDS/dds/core/status/StatusMask.hpp>

namespace dds {
namespace core {
namespace cond {
class StatusCondition;
}

class EntityImpl;

/**
 * @brief This is the abstract base class for all the DDS objects that support
 * QoS policies, a listener and a status condition.
 */
class Entity : public dds::core::Reference<EntityImpl> {
public:
    /**
     * @brief Create the Entity.
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::ERROR
     */
    dds::core::ReturnCode Create(void);

    /**
     * @brief Enable the Entity.
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::ALREADY_DELETED
     */
    dds::core::ReturnCode Enable(void);

    /**
     * @brief Forces the destruction of this entity.
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::ALREADY_DELETED
     */
    dds::core::ReturnCode Close(void);

    /**
     * @brief Retrieves the list of communication statuses that are triggered.
     * @return list of communication statuses in the dds::core::Entity that are
     * triggered.
     */
    dds::core::status::StatusMask GetStatusChanges(void);

    Entity(const Entity& rhs) noexcept;
    Entity(Entity&& rhs) noexcept;
    Entity& operator=(const Entity& rhs) noexcept;
    Entity& operator=(Entity&& rhs) noexcept;

    ~Entity(void) override;
protected:
    Entity(void) noexcept : Reference(nullptr)
    {}

    dds::core::ReturnCode GetAvailability(void) noexcept;

    /**
     * @brief user created entities and their copies have soul, the ones made in callback don't have, when the last soul
     * entity died, sth will happen, for example, unset listeners
     */
    bool isSouled_{true};

    friend class dds::core::cond::StatusCondition;
};
}
}

#endif /* DDS_CORE_ENTITY_HPP */

