/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: EntityId.hpp
 */

#ifndef DDS_CORE_ENTITY_ID_HPP
#define DDS_CORE_ENTITY_ID_HPP

#include <array>

namespace dds {
namespace core {
using EntityKey = std::array<uint8_t, 3U>; /* EntityKey with 3 octets. */

class EntityId {
public:
    void EntityKey(dds::core::EntityKey key) noexcept
    {
        entityKey_ = key;
    }

    dds::core::EntityKey EntityKey() const noexcept
    {
        return entityKey_;
    }

    void EntityKind(uint8_t kind) noexcept
    {
        entityKind_ = kind;
    }

    uint8_t EntityKind() const noexcept
    {
        return entityKind_;
    }

    uint32_t GetEntityId() const noexcept
    {
        return (static_cast<uint32_t>(entityKey_[0U]) << 24U) +  /* Index 0 with ls 24 bits */
            (static_cast<uint32_t>(entityKey_[1U]) << 16U) +     /* Index 1 with ls 16 bits */
            (static_cast<uint32_t>(entityKey_[2U]) << 8U) +      /* Index 2 with ls 8 bits */
            (static_cast<uint32_t>(entityKind_));
    }

    bool operator==(const EntityId &rhs) const noexcept
    {
        return (entityKey_ == rhs.entityKey_) && (entityKind_ == rhs.entityKind_);
    }

private:
    dds::core::EntityKey entityKey_{};
    uint8_t entityKind_{};
};
}
}

#endif /* DDS_CORE_ENTITY_ID_HPP */

