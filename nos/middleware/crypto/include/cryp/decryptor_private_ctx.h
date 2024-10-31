#ifndef ARA_CRYPTO_CRYP_DECRYPTOR_PRIVATE_CTX_H_
#define ARA_CRYPTO_CRYP_DECRYPTOR_PRIVATE_CTX_H_

#include "core/result.h"
#include "cryp/cryobj/private_key.h"
#include "cryp/crypto_service.h"
#include "cryp/crypto_context.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class DecryptorPrivateCtx:public CryptoContext {
public:
    using Uptr = std::unique_ptr<DecryptorPrivateCtx>;
    virtual CryptoService::Uptr GetCryptoService() const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>>ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const PrivateKey& key) noexcept = 0;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_DECRYPTOR_PRIVATE_CTX_H_