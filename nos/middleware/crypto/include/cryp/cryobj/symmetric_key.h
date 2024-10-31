#ifndef ARA_CRYPTO_CRYP_SYMMETRIC_KEY_H_
#define ARA_CRYPTO_CRYP_SYMMETRIC_KEY_H_
#include "restricted_use_object.h"
// #include "imp_crypto_primitive_id.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

class SymmetricKey:public RestrictedUseObject{
public:
    using Uptrc = std::unique_ptr<const SymmetricKey>;
    static const CryptoObjectType kObjectType = CryptoObjectType::kSymmetricKey;
    // SymmetricKey(CryptoPrimitiveId::AlgId algId,AllowedUsageFlags allowedUsage, bool isSession, bool isExportable)
    // : RestrictedUseObject(allowedUsage,isSession,isExportable)
    // , algId_(algId){}

    SymmetricKey(const CryptoObjectInfo& object_info, const CryptoPrimitiveId& primitive_id, const AllowedUsageFlags& usage)
    : RestrictedUseObject(usage, object_info, primitive_id) {

    }

    // SymmetricKey() = default;
    // Usage GetAllowedUsage() const noexcept override;
    // CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
    // COIdentifier GetObjectId () const noexcept override;
    // COIdentifier HasDependence () const noexcept override;
    // std::size_t GetPayloadSize () const noexcept override;
    // bool IsExportable () const noexcept override;
    // bool IsSession () const noexcept override;
    netaos::core::Result<void> Save(IOInterface& container) const noexcept override {
        return netaos::core::Result<void>::FromError(CryptoErrc::kUnsupported);
    }
private:
    CryptoAlgId algId_;
    netaos::core::StringView primitiveName;

//  ImpCryptoPrimitiveId impPrimitiveId_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_SYMMETRIC_KEY_H_