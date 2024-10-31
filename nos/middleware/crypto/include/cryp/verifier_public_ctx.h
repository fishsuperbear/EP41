#ifndef ARA_CRYPTO_CRYP_VERIFIER_PUBLIC_CTX_H_
#define ARA_CRYPTO_CRYP_VERIFIER_PUBLIC_CTX_H_
#include "core/result.h"
#include "common/mem_region.h"
#include "cryp/crypto_context.h"
#include "cryp/signature_service.h"
#include "cryp/verifier_public_ctx.h"
#include "cryp/hash_function_ctx.h"
#include "cryp/cryobj/signature.h"
#include "cryp/cryobj/public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class VerifierPublicCtx:public CryptoContext{
public:

    using Uptr = std::unique_ptr<VerifierPublicCtx>;
    // virtual SignatureService::Uptr GetSignatureService() const noexcept = 0;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const PublicKey& key) noexcept = 0;
    // virtual ara::core::Result<bool> VerifyPrehashed(CryptoAlgId hashAlgId, ReadOnlyMemRegion hashValue, const Signature& signature,
    //                                                 ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;
    virtual netaos::core::Result<bool> Verify(ReadOnlyMemRegion value, ReadOnlyMemRegion signature, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;
    // virtual ara::core::Result<bool> VerifyPrehashed(const HashFunctionCtx& hashFn, const Signature& signature, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;
    // virtual ara::core::Result<bool> VerifyPrehashed(const HashFunctionCtx& hashFn, ReadOnlyMemRegion signature, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;

    private:
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_VERIFIER_PUBLIC_CTX_H_