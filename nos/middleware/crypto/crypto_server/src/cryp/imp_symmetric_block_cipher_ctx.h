#ifndef ARA_CRYPTO_CRYP_IMP_SYMMETRIC_BLOCK_CIPHER_CTX_H_
#define ARA_CRYPTO_CRYP_IMP_SYMMETRIC_BLOCK_CIPHER_CTX_H_

#include <vector>
#include <cstddef>
#include <memory>
#include <openssl/err.h>
#include <openssl/evp.h>
// #include <openssl/types.h>
#include <core/result.h>

#include "common/base_id_types.h"
#include "common/mem_region.h"
#include "cryp/cryobj/simpl_symmetric_key.h"
#include "cryp/symmetric_block_cipher_ctx.h"
#include "cryp/crypto_service.h"
#include "cryp/crypto_context.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpSymmetricBlockCipherCtx:public SymmetricBlockCipherCtx{
public:
    using Uptr = std::unique_ptr<ImpSymmetricBlockCipherCtx>;
    ImpSymmetricBlockCipherCtx(CryptoAlgId alg_id);
    CryptoService::Uptr GetCryptoService() const noexcept override;
    netaos::core::Result<CryptoTransform> GetTransformation() const noexcept override;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    netaos::core::Result<netaos::core::Vector<uint8_t>> ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept override;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept;
    netaos::core::Result<netaos::core::Vector<uint8_t>> ProcessBlocks(ReadOnlyMemRegion in) const noexcept override;
    netaos::core::Result<void> Reset() noexcept override;
    netaos::core::Result<void> SetKey(const SymmetricKey& key, CryptoTransform transform = CryptoTransform::kEncrypt) noexcept override; 
    CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
    bool IsInitialized () const noexcept override;
    CryptoProvider& MyProvider() const noexcept override;
private:
    CryptoAlgId alg_id_ = kAlgIdUndefined;
    bool isInitialized_ = false;
    CryptoTransform transform_;
    std::unique_ptr<SimplSymmetricKey> symmetric_key_;
    const EVP_CIPHER *cipher_ = nullptr;
    EVP_CIPHER_CTX*  ctx_ = nullptr;
    mutable std::vector<std::uint8_t> out_;
    mutable int outlen_ = 0;

    const EVP_CIPHER *getEvpCipher();

};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_IMP_SYMMETRIC_BLOCK_CIPHER_CTX_H_