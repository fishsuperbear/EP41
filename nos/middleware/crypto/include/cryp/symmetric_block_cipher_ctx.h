#ifndef ARA_CRYPTO_CRYP_SYMMETRIC_BLOCK_CIPHER_CTX_H_
#define ARA_CRYPTO_CRYP_SYMMETRIC_BLOCK_CIPHER_CTX_H_

#include <vector>
#include <cstddef>
#include "core/result.h"
#include "common/base_id_types.h"
#include "common/mem_region.h"
#include "cryp/crypto_context.h"
#include "cryp/crypto_service.h"
#include "cryp/cryobj/symmetric_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class SymmetricBlockCipherCtx : public CryptoContext {
public:
    using Uptr = std::unique_ptr<SymmetricBlockCipherCtx>;
    virtual CryptoService::Uptr GetCryptoService() const noexcept = 0;
    virtual netaos::core::Result<CryptoTransform> GetTransformation() const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte>> ProcessBlocks(ReadOnlyMemRegion in) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> ProcessBlocks(ReadOnlyMemRegion in) const noexcept = 0;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const SymmetricKey& key, CryptoTransform transform = CryptoTransform::kEncrypt) noexcept = 0;   private:
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_SYMMETRIC_BLOCK_CIPHER_CTX_H_