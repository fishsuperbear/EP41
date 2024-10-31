#ifndef ARA_CRYPTO_CRYP_KEY_DERIVATION_FUNCTION_CTX_H_
#define ARA_CRYPTO_CRYP_KEY_DERIVATION_FUNCTION_CTX_H_
#include "core/result.h"
#include "cryp/crypto_context.h"
#include "common/mem_region.h"
#include "common/base_id_types.h"
#include "common/io_interface.h"
#include "common/serializable.h"
#include "common/volatile_trusted_container.h"
#include "cryp/cryobj/symmetric_key.h"
#include "cryp/cryobj/private_key.h"
#include "cryp/cryobj/crypto_primitive_id.h"
#include "cryp/cryobj/crypto_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class KeyDerivationFunctionCtx:public CryptoContext{
public:
    using Uptr = std::unique_ptr<KeyDerivationFunctionCtx>;
    virtual netaos::core::Result<void> AddSalt(ReadOnlyMemRegion salt) noexcept = 0;
    virtual netaos::core::Result<void> AddSecretSalt(const SecretSeed& salt) noexcept = 0;
    virtual std::uint32_t ConfigIterations(std::uint32_t iterations = 0) noexcept = 0;
    virtual netaos::core::Result<SymmetricKey::Uptrc> DeriveKey(bool isSession = true, bool isExportable = false) const noexcept = 0;
    virtual netaos::core::Result<SecretSeed::Uptrc> DeriveSeed(bool isSession = true, bool isExportable = false) const noexcept = 0;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual ExtensionService::Uptr GetExtensionService() const noexcept = 0;
    virtual std::size_t GetKeyIdSize () const noexcept=0;
    virtual AlgId GetTargetAlgId() const noexcept = 0;
    virtual AllowedUsageFlags GetTargetAllowedUsage () const noexcept=0;
    virtual std::size_t GetTargetKeyBitLength() const noexcept = 0;
    virtual netaos::core::Result<void> Init(ReadOnlyMemRegion targetKeyId, AlgId targetAlgId = kAlgIdAny, AllowedUsageFlags allowedUsage = kAllowKdfMaterialAnyUsage,
                                         ReadOnlyMemRegion ctxLabel = ReadOnlyMemRegion()) noexcept = 0;
    virtual netaos::core::Result<void> SetSourceKeyMaterial(const RestrictedUseObject& sourceKM) noexcept = 0;


    private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_KEY_DERIVATION_FUNCTION_CTX_H_