#ifndef ARA_CRYPTO_CRYP_KEY_ENCAPSULATOR_PUBLIC_CTX_H_
#define ARA_CRYPTO_CRYP_KEY_ENCAPSULATOR_PUBLIC_CTX_H_

#include <vector>
#include "core/result.h"
#include "common/base_id_types.h"
#include "cryp/cryobj/restricted_use_object.h"
#include "cryp/key_derivation_function_ctx.h"
#include "cryp/extension_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class KeyEncapsulatorPublicCtx:public CryptoContext{
public:
    using Uptr = std::unique_ptr<KeyEncapsulatorPublicCtx>;
    virtual std::size_t GetEncapsulatedSize() const noexcept = 0;
    virtual ExtensionService::Uptr GetExtensionService() const noexcept = 0;
    virtual std::size_t GetKekEntropy() const noexcept = 0;
    virtual netaos::core::Result<void> AddKeyingData(const RestrictedUseObject& keyingData) noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > Encapsulate(KeyDerivationFunctionCtx& kdf, CryptoAlgId kekAlgId) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>>Encapsulate(KeyDerivationFunctionCtx& kdf, CryptoAlgId kekAlgId) const noexcept = 0;
    virtual netaos::core::Result<void> Reset() noexcept = 0;

   private:
   
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_KEY_ENCAPSULATOR_PUBLIC_CTX_H_