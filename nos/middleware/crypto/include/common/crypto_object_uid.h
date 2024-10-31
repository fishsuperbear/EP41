#ifndef ARA_CRYPTO_COMMON_CRYPTO_OBJECT_UID_H_
#define ARA_CRYPTO_COMMON_CRYPTO_OBJECT_UID_H_
#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "uuid.h"

namespace hozon {
namespace netaos {
namespace crypto {

struct CryptoObjectUid {
    Uuid mGeneratorUid;
    std::uint64_t mVersionStamp = 0u;
    bool IsNil () const noexcept;
    bool SourceIsNil () const noexcept;
    constexpr bool HasEarlierVersionThan(const CryptoObjectUid& anotherId) const noexcept;
    constexpr bool HasLaterVersionThan(const CryptoObjectUid& anotherId) const noexcept;
    constexpr bool HasSameSourceAs(const CryptoObjectUid& anotherId) const noexcept;
    // constexpr bool operator==(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept;
    // constexpr bool operator<(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept;
    // constexpr bool operator>(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept;
    // constexpr bool operator!=(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept;
    // constexpr bool operator<=(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept;
    // constexpr bool operator>=(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept;
};

#define CRYPTO_OBJECT_UID_INITIALIZER CryptoObjectUid { Uuid {0, 0}, 4 }

inline bool CryptoObjectUid::IsNil() const noexcept
{
    return mGeneratorUid.IsNil();
}

inline constexpr bool CryptoObjectUid::HasSameSourceAs(const CryptoObjectUid& anotherId) const noexcept
{
    return mGeneratorUid == anotherId.mGeneratorUid;
}

inline constexpr bool CryptoObjectUid::HasEarlierVersionThan(const CryptoObjectUid& anotherId) const noexcept
{
    return (mGeneratorUid < anotherId.mGeneratorUid);
}

inline constexpr bool CryptoObjectUid::HasLaterVersionThan(const CryptoObjectUid& anotherId) const noexcept
{
    return (mGeneratorUid > anotherId.mGeneratorUid);
}

inline constexpr bool operator==(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept
{
    return lhs.HasSameSourceAs(rhs);
}

inline constexpr bool operator!=(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept
{
    return !(lhs.mGeneratorUid == rhs.mGeneratorUid);
}

constexpr bool operator<(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept
{
    return (lhs.mGeneratorUid < rhs.mGeneratorUid);
}

constexpr bool operator>(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept
{
    return (lhs.mGeneratorUid > rhs.mGeneratorUid);
}

constexpr bool operator<=(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept
{
    return (lhs.mGeneratorUid <= rhs.mGeneratorUid);
}

constexpr bool operator>=(const CryptoObjectUid& lhs, const CryptoObjectUid& rhs) noexcept
{
    return (lhs.mGeneratorUid >= rhs.mGeneratorUid);
}

}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_COMMON_CRYPTO_OBJECT_UID_H_