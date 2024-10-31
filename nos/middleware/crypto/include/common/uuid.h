#pragma once

#include <utility>
#include <cstdint>
#include <string>

#include "core/result.h"

namespace hozon {
namespace netaos {
namespace crypto {

struct Uuid {
    std::uint64_t mQwordLs = 0u;
    std::uint64_t mQwordMs = 0u;
    bool IsNil () const noexcept;
    Uuid() = default;
    Uuid(std::uint64_t ls,std::uint64_t ms):mQwordLs(ls),mQwordMs(ms){}

    std::string ToUuidStr() const noexcept;
};

inline bool Uuid::IsNil() const noexcept
{
    return (mQwordLs == 0U) && (mQwordMs == 0U);
}

inline constexpr bool operator<(const Uuid& lhs, const Uuid& rhs) noexcept
{
    return std::make_pair(lhs.mQwordLs, lhs.mQwordMs) < std::make_pair(rhs.mQwordLs, rhs.mQwordMs);
}

inline constexpr bool operator>(const Uuid& lhs, const Uuid& rhs) noexcept
{
    return std::make_pair(lhs.mQwordLs, lhs.mQwordMs) > std::make_pair(rhs.mQwordLs, rhs.mQwordMs);
}

inline constexpr bool operator==(const Uuid& lhs, const Uuid& rhs) noexcept
{
        return std::make_pair(lhs.mQwordLs, lhs.mQwordMs) == std::make_pair(rhs.mQwordLs, rhs.mQwordMs);
}

inline constexpr bool operator!=(const Uuid& lhs, const Uuid& rhs) noexcept
{
    return !(lhs == rhs);
}

inline constexpr bool operator<=(const Uuid& lhs, const Uuid& rhs) noexcept
{
    return !(lhs > rhs);
}

inline constexpr bool operator>=(const Uuid& lhs, const Uuid& rhs) noexcept
{
    return !(lhs < rhs);
}

netaos::core::Result<Uuid> FromString(const std::string strUuid) noexcept;

netaos::core::Result<Uuid> MakeVersion4Uuid() noexcept;

}  // namespace crypto
}  // namespace netaos
}  // namespace hozon