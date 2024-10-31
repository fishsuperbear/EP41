#ifndef ARA_CRYPTO_CRYP_ENCRYPTOR_PUBLIC_CTX_H_
#define ARA_CRYPTO_CRYP_ENCRYPTOR_PUBLIC_CTX_H_
#include <cstddef>
#include "core/result.h"
#include "core/vector.h"
#include "common/mem_region.h"
#include "cryp/crypto_context.h"
#include "cryp/encryptor_public_ctx.h"
#include "cryp/crypto_service.h"
#include "cryp/cryobj/public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class EncryptorPublicCtx:public CryptoContext {
public:
    using Uptr = std::unique_ptr<EncryptorPublicCtx>;
    virtual CryptoService::Uptr GetCryptoService() const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>>ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const PublicKey& key) noexcept = 0;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_ENCRYPTOR_PUBLIC_CTX_H_