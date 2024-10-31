#ifndef ARA_CRYPTO_CRYP_IMP_DECRYPTOR_PRIVATE_CTX_H_
#define ARA_CRYPTO_CRYP_IMP_DECRYPTOR_PRIVATE_CTX_H_

#include "openssl/ossl_typ.h"

#include "core/result.h"
#include "common/mem_region.h"
#include "cryp/decryptor_private_ctx.h"
#include "cryp/cryobj/private_key.h"
#include "cryp/crypto_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpDecryptorPrivateCtx:public DecryptorPrivateCtx {
public:
    using Uptr = std::unique_ptr<ImpDecryptorPrivateCtx>;
    CryptoService::Uptr GetCryptoService() const noexcept override;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept = 0;
    netaos::core::Result<netaos::core::Vector<uint8_t>>ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept override;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding = false) const noexcept;
    netaos::core::Result<void> Reset() noexcept override;
    netaos::core::Result<void> SetKey(const PrivateKey& key) noexcept override;
    CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
    bool IsInitialized () const noexcept override;
    CryptoProvider& MyProvider() const noexcept override;

private:
    CryptoAlgId alg_id_ = kAlgIdUndefined;
    bool isInitialized_ = false;
    EVP_PKEY_CTX *pkey_ctx_ = nullptr;
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_DECRYPTOR_PRIVATE_CTX_H_