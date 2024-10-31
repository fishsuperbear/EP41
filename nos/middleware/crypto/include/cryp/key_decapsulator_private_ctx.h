#ifndef ARA_CRYPTO_CRYP_KEY_DECAPSULATOR_PRIVATE_CTX_H_
#define ARA_CRYPTO_CRYP_KEY_DECAPSULATOR_PRIVATE_CTX_H_
#include "core/result.h"
#include "core/optional.h"
#include "cryp/cryobj/symmetric_key.h"
#include "cryp/cryobj/secret_seed.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class KeyDecapsulatorPrivateCtx:public CryptoContext{
public:

    using Uptr = std::unique_ptr<KeyDecapsulatorPrivateCtx>;
    virtual netaos::core::Result<SymmetricKey::Uptrc> DecapsulateKey(ReadOnlyMemRegion input, CryptoAlgId keyingDataAlgId, KeyDerivationFunctionCtx& kdf, CryptoAlgId kekAlgId,
                                                               netaos::core::Optional<AllowedUsageFlags> allowedUsage) const noexcept = 0;
    virtual netaos::core::Result<SecretSeed::Uptrc> DecapsulateSeed(ReadOnlyMemRegion input, netaos::core::Optional<AllowedUsageFlags> allowedUsage) const noexcept = 0;
    virtual std::size_t GetEncapsulatedSize() const noexcept = 0;
    virtual ExtensionService::Uptr GetExtensionService() const noexcept = 0;
    virtual std::size_t GetKekEntropy() const noexcept = 0;
    virtual netaos::core::Result<void> Reset () noexcept=0;
    virtual netaos::core::Result<void> SetKey(const PrivateKey& key) noexcept = 0;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_KEY_DECAPSULATOR_PRIVATE_CTX_H_