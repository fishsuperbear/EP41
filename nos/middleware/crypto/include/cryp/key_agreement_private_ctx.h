#ifndef ARA_CRYPTO_CRYP_KEY_AGREEMENT_PRIVATE_CTX_H_
#define ARA_CRYPTO_CRYP_KEY_AGREEMENT_PRIVATE_CTX_H_
#include "core/result.h"
#include "core/optional.h"
#include "crypto_context.h"
#include "extension_service.h"
#include "extension_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class KeyAgreementPrivateCtx:public CryptoContext{
public:
    using Uptr = std::unique_ptr<KeyAgreementPrivateCtx>;
    // virtual ara::core::Result<SymmetricKey::Uptrc> AgreeKey(const Public Key& otherSideKey, CryptoAlgId targetAlgId, AllowedUsageFlags allowed Usage,
    //                                                      ara::core::Optional<const KeyDerivationFunctionCtx::Uptr> kdf, ara::core::Optional<ReadOnlyMemRegion> salt,
    //                                                      ara::core::Optional<ReadOnlyMemRegion> ctxLabel) const noexcept = 0;
    virtual netaos::core::Result<SecretSeed::Uptrc> AgreeSeed(const PublicKey& otherSideKey, netaos::core::Optional<AllowedUsageFlags> allowedUsage) const noexcept = 0;
    virtual ExtensionService::Uptr GetExtensionService() const noexcept = 0;
    virtual netaos::core::Result<void> Reset() noexcept=0;
private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_KEY_AGREEMENT_PRIVATE_CTX_H_