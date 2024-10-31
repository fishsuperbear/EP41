#ifndef ARA_CRYPTO_KEYS_KEY_SLOT_CONTENT_PROPS_H_
#define ARA_CRYPTO_KEYS_KEY_SLOT_CONTENT_PROPS_H_
#include "common/base_id_types.h"
#include "common/crypto_object_uid.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {


struct KeySlotContentProps {
    using Uptr = std::unique_ptr<KeySlotContentProps>;
    // KeySlotContentProps() = default;
    CryptoAlgId mAlgId;
    std::size_t mObjectSize;
    CryptoObjectType mObjectType;
    CryptoObjectUid mObjectUid;
    AllowedUsageFlags mContentAllowedUsage;

    KeySlotContentProps()
    : mAlgId(hozon::netaos::crypto::kAlgIdUndefined)
    , mObjectSize(0)
    , mObjectType(hozon::netaos::crypto::CryptoObjectType::kUndefined)
    , mContentAllowedUsage(hozon::netaos::crypto::kAllowPrototypedOnly) {

    }
};


inline constexpr bool operator==(const KeySlotContentProps& lhs, const KeySlotContentProps& rhs) noexcept{
    return (lhs.mObjectUid == rhs.mObjectUid) && (lhs.mAlgId == rhs.mAlgId) &&
        (lhs.mObjectSize == rhs.mObjectSize) && (lhs.mContentAllowedUsage == rhs.mContentAllowedUsage) &&
        (lhs.mObjectType == rhs.mObjectType);
}

inline constexpr bool operator!=(const KeySlotContentProps& lhs, const KeySlotContentProps& rhs) noexcept{
    return !(lhs == rhs);
}

}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_KEYS_KEY_SLOT_CONTENT_PROPS_H_