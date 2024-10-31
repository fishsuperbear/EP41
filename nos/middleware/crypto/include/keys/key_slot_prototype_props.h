#ifndef ARA_CRYPTO_KEYS_KEY_SLOT_PROTOTYPE_PROPS_H_
#define ARA_CRYPTO_KEYS_KEY_SLOT_PROTOTYPE_PROPS_H_

#include "core/result.h"
#include "common/uuid.h"
#include "common/base_id_types.h"
#include "common/crypto_object_uid.h"
#include "keys/elementary_types.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {
struct KeySlotPrototypeProps {
    using Uptr = std::unique_ptr<KeySlotPrototypeProps>;
    // KeySlotPrototypeProps() = default;

    CryptoAlgId mAlgId;
    bool mAllocateSpareSlot;
    bool mAllowContentTypeChange;
    AllowedUsageFlags mContentAllowedUsage;
    bool mExportAllowed;
    std::int32_t mMaxUpdateAllowed;
    KeySlotType mSlotType;
    std::size_t mSlotCapacity;
    CryptoObjectType mObjectType;
    std::uint32_t mslotNum; //add by neta
    crypto::Uuid mslotUuid; //add by neta

    KeySlotPrototypeProps()
    : mAlgId(hozon::netaos::crypto::kAlgIdUndefined)
    , mAllocateSpareSlot(false)
    , mAllowContentTypeChange(false)
    , mContentAllowedUsage(hozon::netaos::crypto::kAllowPrototypedOnly)
    , mExportAllowed(false)
    , mMaxUpdateAllowed(0)
    , mSlotType(hozon::netaos::crypto::KeySlotType::kUnknown)
    , mSlotCapacity(0)
    , mObjectType(hozon::netaos::crypto::CryptoObjectType::kUndefined)
    , mslotNum(0)
    , mslotUuid() {

    }
};


inline constexpr bool operator==(const KeySlotPrototypeProps& lhs, const KeySlotPrototypeProps& rhs) noexcept{
    return (lhs.mAlgId == rhs.mAlgId) && (lhs.mAllocateSpareSlot == rhs.mAllocateSpareSlot) &&
        (lhs.mAllowContentTypeChange == rhs.mAllowContentTypeChange) && (lhs.mContentAllowedUsage == rhs.mContentAllowedUsage) &&
        (lhs.mExportAllowed == rhs.mExportAllowed) && (lhs.mSlotType == rhs.mSlotType) && (lhs.mSlotCapacity == rhs.mSlotCapacity)
        && (lhs.mObjectType == rhs.mObjectType)  && (lhs.mslotNum == rhs.mslotNum);
}

inline constexpr bool operator!=(const KeySlotPrototypeProps& lhs, const KeySlotPrototypeProps& rhs) noexcept{
    return !(lhs == rhs);
}

}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_KEY_SLOT_PROTOTYPE_PROPS_H_