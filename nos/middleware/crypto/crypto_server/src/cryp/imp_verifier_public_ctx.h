#ifndef ARA_CRYPTO_CRYP_IMP_VERIFIER_PUBLIC_CTX_H_
#define ARA_CRYPTO_CRYP_IMP_VERIFIER_PUBLIC_CTX_H_

#include <cstddef>
#include <vector>
#include <utility>
#include <string>
#include <cstdio>

#include "core/result.h"
#include "common/mem_region.h"
#include "cryp/crypto_context.h"
#include "cryp/signature_service.h"
#include "cryp/verifier_public_ctx.h"
#include "cryp/cryobj/signature.h"
#include "cryp/hash_function_ctx.h"
#include "cryp/cryobj/imp_public_key.h"
#include "cryp/cryobj/public_key.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "openssl/err.h"
#include "openssl/evp.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpVerifierPublicCtx:public VerifierPublicCtx{
public:
    struct OpensslCtx
    {
        OSSL_LIB_CTX* plib_ctx;
        EVP_MD* pmd;
        EVP_MD_CTX* pmd_ctx;
        unsigned char* pverify_value;
        unsigned int verify_len;
        OpensslCtx():plib_ctx(NULL),pmd(NULL),pmd_ctx(NULL),pverify_value(NULL),verify_len(0){}
    };
    using Uptr = std::unique_ptr<ImpVerifierPublicCtx>;
    ImpVerifierPublicCtx(AlgId id):alg_id_(id){}
    // SignatureService::Uptr GetSignatureService() const noexcept override;
    netaos::core::Result<void> Reset() noexcept override;
    netaos::core::Result<void> SetKey(const PublicKey& key) noexcept override;
    // ara::core::Result<bool> VerifyPrehashed(CryptoAlgId hashAlgId, ReadOnlyMemRegion hashValue, const Signature& signature,
    //                                                 ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept override;
    netaos::core::Result<bool> Verify(ReadOnlyMemRegion value, ReadOnlyMemRegion signature, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept override;
    // ara::core::Result<bool> VerifyPrehashed(const HashFunctionCtx& hashFn, const Signature& signature, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept override;
    // ara::core::Result<bool> VerifyPrehashed(const HashFunctionCtx& hashFn, ReadOnlyMemRegion signature, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept override;
    CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
    bool IsInitialized () const noexcept override;
    CryptoProvider& MyProvider() const noexcept override;
    int dump_key(const EVP_PKEY* pkey);//TODO for test
private:
    CryptoAlgId alg_id_ = kAlgIdUndefined;
    bool isInitialized_ = false;
    mutable OpensslCtx openssl_ctx_;
    // PublicKey *ppublic_key_;
    ImpPublicKey *ppublic_key_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#ifdef __cplusplus
}
#endif

#endif  // #define ARA_CRYPTO_CRYP_IMP_VERIFIER_PUBLIC_CTX_H_