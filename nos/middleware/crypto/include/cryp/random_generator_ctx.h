#ifndef ARA_CRYPTO_CRYP_RANDOM_GENERATOR_CTX_H_
#define ARA_CRYPTO_CRYP_RANDOM_GENERATOR_CTX_H_

#include "core/result.h"
#include "cryp/crypto_context.h"
#include "cryp/extension_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class RandomGeneratorCtx : public CryptoContext{
public:
    using Uptr = std::unique_ptr<RandomGeneratorCtx>;
    virtual bool AddEntropy (ReadOnlyMemRegion entropy) noexcept=0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > Generate(std::uint32_t count) noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t> > Generate(std::uint32_t count) noexcept = 0;
    virtual ExtensionService::Uptr GetExtensionService() const noexcept = 0;
    virtual bool Seed(ReadOnlyMemRegion seed) noexcept = 0;
    virtual bool Seed(const SecretSeed& seed) noexcept = 0;
    virtual bool SetKey(const SymmetricKey& key) noexcept = 0;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_RANDOM_GENERATOR_CTX_H_