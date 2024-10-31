#ifndef ARA_CRYPTO_CRYP_PRIVATE_KEY_H_
#define ARA_CRYPTO_CRYP_PRIVATE_KEY_H_

#include "core/result.h"
#include "public_key.h"
#include "restricted_use_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class PrivateKey:public RestrictedUseObject{
public:
    using Uptrc = std::unique_ptr<const PrivateKey>;
    // PrivateKey(AllowedUsageFlags allowedUsage, bool isSession, bool isExportable):RestrictedUseObject(allowedUsage,isSession,isExportable){}
    PrivateKey(const CryptoObjectInfo& object_info, const CryptoPrimitiveId& primitive_id, const AllowedUsageFlags& usage)
    : RestrictedUseObject(usage, object_info, primitive_id) {

    }
    virtual netaos::core::Result<PublicKey::Uptrc> GetPublicKey() const noexcept = 0;
    virtual bool CheckKey(bool strongCheck = true) const noexcept = 0;
    static const CryptoObjectType kObjectType = CryptoObjectType::kPrivateKey;
private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_PRIVATE_KEY_H_