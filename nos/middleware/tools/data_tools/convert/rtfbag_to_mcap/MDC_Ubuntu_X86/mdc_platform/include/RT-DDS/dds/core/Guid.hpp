/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Guid.hpp
 */

#ifndef DDS_CORE_GUID_HPP
#define DDS_CORE_GUID_HPP

#include <string>
#include <sstream>
#include <iomanip>
#include <RT-DDS/dds/core/EntityId.hpp>

namespace dds {
namespace core {
using GuidPrefix = std::array<uint8_t, 12U>;  /* GuidPrefix with 12 octets. */

class Guid {
public:
    void GuidPrefix(dds::core::GuidPrefix prefix) noexcept
    {
        guidPrefix_ = prefix;
    }

    dds::core::GuidPrefix GuidPrefix() const noexcept
    {
        return guidPrefix_;
    }

    void EntityId(dds::core::EntityId id) noexcept
    {
        entityId_ = id;
    }

    dds::core::EntityId EntityId() const noexcept
    {
        return entityId_;
    }

    bool operator==(const Guid &rhs) const noexcept
    {
        return (guidPrefix_ == rhs.guidPrefix_) && (entityId_ == rhs.entityId_);
    }

    static std::string ConvertGUIDToString(const Guid &guid)
    {
        std::string guidStr = "";
        std::stringstream guidSS;
        guidSS << "("
               << GetGuidPrefix1(guid.GuidPrefix())
               << ",0x"
               << std::hex << std::setw(8) << std::setfill('0')  /* 8: digit capacity */
               << GetGuidPrefix2(guid.GuidPrefix())
               << ","
               << std::dec << GetGuidPrefix3(guid.GuidPrefix())
               << ",0x"
               << std::hex << std::setw(8) << std::setfill('0')  /* 8: digit capacity */
               << guid.EntityId().GetEntityId()
               << ")";
        guidSS >> guidStr;
        return guidStr;
    }

private:


    static uint32_t GetGuidPrefix1(const dds::core::GuidPrefix &guidPrefix) noexcept
    {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return (static_cast<uint32_t>(guidPrefix[2U])) +        /* Index 2 with ls 0 bits */
               (static_cast<uint32_t>(guidPrefix[3U]) << 8U);       /* Index 3 with ls 8 bits */


#else
        return (static_cast<uint32_t>(guidPrefix[2]) << 8) +   /* Index 2 with ls 8 bits */
            (static_cast<uint32_t>(guidPrefix[3]));            /* Index 3 with ls 0 bits */


#endif
    }

    static uint32_t GetGuidPrefix2(const dds::core::GuidPrefix &guidPrefix) noexcept
    {
        return (static_cast<uint32_t>(guidPrefix[4U]) << 24U) +  /* Index 4 with ls 24 bits */
            (static_cast<uint32_t>(guidPrefix[5U]) << 16U) +     /* Index 5 with ls 16 bits */
            (static_cast<uint32_t>(guidPrefix[6U]) << 8U) +      /* Index 6 with ls 8 bits */
            (static_cast<uint32_t>(guidPrefix[7U]));            /* Index 7 with ls 0 bits */
    }

    static uint32_t GetGuidPrefix3(const dds::core::GuidPrefix &guidPrefix) noexcept
    {
        return (static_cast<uint32_t>(guidPrefix[8U])) +       /* Index 8 with ls 0 bits */
            (static_cast<uint32_t>(guidPrefix[9U]) << 8U) +     /* Index 9 with ls 8 bits */
            (static_cast<uint32_t>(guidPrefix[10U]) << 16U) +   /* Index 10 with ls 16 bits */
            (static_cast<uint32_t>(guidPrefix[11U]) << 24U);    /* Index 11 with ls 24 bits */
    }

    dds::core::GuidPrefix guidPrefix_{};
    dds::core::EntityId entityId_{};
};
}
}

#endif /* DDS_CORE_GUID_HPP */

