#ifndef ARA_CRYPTO_CRYP_RESTRICTED_USE_OBJECT_H_
#define ARA_CRYPTO_CRYP_RESTRICTED_USE_OBJECT_H_
#include "crypto_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class RestrictedUseObject:public CryptoObject{
public:
    using Uptrc = std::unique_ptr<const RestrictedUseObject>;
    using Usage = AllowedUsageFlags;

    RestrictedUseObject(const Usage& usage, const CryptoObjectInfo& crypto_object_info, const CryptoPrimitiveId& primitive_id)
    : CryptoObject(crypto_object_info, primitive_id)
    , allowed_usage_(usage) {

    }

    virtual Usage GetAllowedUsage() const noexcept {
        return allowed_usage_;
    }

    // RestrictedUseObject() = default;
    virtual ~RestrictedUseObject() = default;
protected:
    Usage allowed_usage_;
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_RESTRICTED_USE_OBJECT_H_