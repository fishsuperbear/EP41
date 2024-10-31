#ifndef ARA_CRYPTO_CRYP_SYMMETRIC_KEY_WRAPPER_CTX_H_
#define ARA_CRYPTO_CRYP_SYMMETRIC_KEY_WRAPPER_CTX_H_

#include <vector>
#include <cstddef>
#include "core/result.h"
#include "common/base_id_types.h"
#include "common/mem_region.h"
#include "cryp/crypto_context.h"
#include "cryp/cryobj/restricted_use_object.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class SymmetricKeyWrapperCtx:public CryptoContext{
public:
    using Uptr = std::unique_ptr<SymmetricKeyWrapperCtx>;
    virtual std::size_t CalculateWrappedKeySize(std::size_t keyLength) const noexcept = 0;
    virtual ExtensionService::Uptr GetExtensionService() const noexcept = 0;
    virtual std::size_t GetMaxTargetKeyLength() const noexcept = 0;
    virtual std::size_t GetTargetKeyGranularity() const noexcept = 0;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const SymmetricKey& key, CryptoTransform transform) noexcept = 0;
    template <typename ExpectedKey>
    netaos::core::Result<typename ExpectedKey::Uptrc> UnwrapConcreteKey(ReadOnlyMemRegion wrappedKey, AlgId algId, AllowedUsageFlags allowedUsage) noexcept;
    virtual netaos::core::Result<RestrictedUseObject::Uptrc> UnwrapKey(ReadOnlyMemRegion wrappedKey, AlgId algId, AllowedUsageFlags allowedUsage) const noexcept = 0;
    virtual netaos::core::Result<SecretSeed::Uptrc> UnwrapSeed(ReadOnlyMemRegion wrappedSeed, AlgId targetAlgId, SecretSeed::Usage allowedUsage) const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > WrapKeyMaterial(const RestrictedUseObject& key) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t> > WrapKeyMaterial(const RestrictedUseObject& key) const noexcept = 0;
   private:
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_SYMMETRIC_KEY_WRAPPER_CTX_H_