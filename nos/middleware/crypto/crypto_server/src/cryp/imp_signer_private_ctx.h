#ifndef ARA_CRYPTO_CRYP_IMP_SIGNER_PRIVATE_CTX_H_
#define ARA_CRYPTO_CRYP_IMP_SIGNER_PRIVATE_CTX_H_

#include <cstddef>
#include <vector>
#include <utility>
#include <string>
#include <cstdio>
#include "cryp/signer_private_ctx.h"
#include "cryp/cryobj/private_key.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "openssl/err.h"
#include "openssl/evp.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpSignerPrivateCtx:public SignerPrivateCtx{
public:
    struct OpensslCtx
    {
        OSSL_LIB_CTX* plib_ctx;
        EVP_MD* pmd;
        EVP_MD_CTX* pmd_ctx;
        // unsigned char* psign_value;
        // unsigned int sign_len;
        // OpensslCtx():plib_ctx(NULL),pmd(NULL),pmd_ctx(NULL),psign_value(NULL),sign_len(0){}
        OpensslCtx():plib_ctx(NULL),pmd(NULL),pmd_ctx(NULL){}
    };

    using Uptr = std::unique_ptr<ImpSignerPrivateCtx>;
    ImpSignerPrivateCtx(AlgId id):alg_id_(id){}

    // SignatureService::Uptr GetSignatureService() const noexcept override;
    netaos::core::Result<void> Reset() noexcept override;
    netaos::core::Result<void> SetKey(const PrivateKey& key) noexcept override;
    // ara::core::Result<Signature::Uptrc> SignPreHashed (const HashFunctionCtx &hashFn, ReadOnlyMemRegion context=ReadOnlyMemRegion()) const noexcept override;
    // ara::core::Result<Signature::Uptrc> SignPreHashed(AlgId hashAlgId, ReadOnlyMemRegion hashValue, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept override;
    netaos::core::Result<netaos::core::Vector<uint8_t>> SignPreHashed (const HashFunctionCtx &hashFn, ReadOnlyMemRegion context=ReadOnlyMemRegion()) const noexcept override;
    netaos::core::Result<netaos::core::Vector<uint8_t>> SignPreHashed(AlgId hashAlgId, ReadOnlyMemRegion hashValue, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept override;
    // ara::core::Result<ara::core::Vector<ara::core::Byte> > Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;
    netaos::core::Result<netaos::core::Vector<uint8_t>>Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept override;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept;

    netaos::core::Result<PublicKey::Uptrc> GetPublicKey() const noexcept override; // add by neta
    CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
    bool IsInitialized () const noexcept override;
    CryptoProvider& MyProvider() const noexcept override;
    PrivateKey* GetPrivateKey() {
        return pprivate_key_;
    }
private:
    CryptoAlgId alg_id_ = kAlgIdUndefined;
    bool isInitialized_ = false;
    PrivateKey *pprivate_key_;
    mutable OpensslCtx openssl_ctx_;
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#ifdef __cplusplus
}
#endif

#endif  // #define ARA_CRYPTO_CRYP_IMP_SIGNER_PRIVATE_CTX_H_