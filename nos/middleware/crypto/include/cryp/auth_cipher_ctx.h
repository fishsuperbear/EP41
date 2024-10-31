#ifndef ARA_CRYPTO_CRYP_AUTH_CIPHER_CTX_H_
#define ARA_CRYPTO_CRYP_AUTH_CIPHER_CTX_H_

#include <cstddef>
#include "core/optional.h"
#include "core/result.h"
#include "crypto_context.h"
#include "mem_region.h"
#include "base_id_types.h"
#include "restricted_use_object.h"
#include "signature.h"
#include "digest_service.h"
#include "symmetric_key.h"
#include "secret_seed.h"
#include "crypto_primitive_id.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class AuthCipherCtx:public CryptoContext{
public:
    using Uptr = std::unique_ptr<AuthCipherCtx>;
    virtual ~AuthCipherCtx() = default;

    virtual netaos::core::Result<bool> Check (const Signature &expected) const noexcept=0;
    virtual DigestService::Uptr GetDigestService () const noexcept=0;
    // template <typename Alloc = <implementation-defined>>
    // ara::core::Result<ByteVector<Alloc> > GetDigest (std::size_t offset=0) const noexcept;
    virtual netaos::core::Result<CryptoTransform> GetTransformation () const noexcept=0;
    virtual std::uint64_t GetMaxAssociatedDataSize () const noexcept=0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > ProcessConfidentialData (ReadOnlyMemRegion in, ara::core::Optional< ReadOnlyMemRegion > expectedTag) noexcept=0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t> > ProcessConfidentialData (ReadOnlyMemRegion in, netaos::core::Optional< ReadOnlyMemRegion > expectedTag) noexcept=0;
    virtual netaos::core::Result<void> ProcessConfidentialData (ReadWriteMemRegion inOut, netaos::core::Optional< ReadOnlyMemRegion > expectedTag) noexcept=0;
    virtual netaos::core::Result<void> SetKey (const SymmetricKey &key,CryptoTransform transform=CryptoTransform::kEncrypt) noexcept=0;
    virtual netaos::core::Result<void> Start (ReadOnlyMemRegion iv=ReadOnlyMemRegion()) noexcept=0;
    virtual netaos::core::Result<void> Start (const SecretSeed &iv)noexcept=0;
    virtual netaos::core::Result<void> UpdateAssociatedData (const RestrictedUseObject &in) noexcept=0;
    virtual netaos::core::Result<void> UpdateAssociatedData (ReadOnlyMemRegion in) noexcept=0;
    virtual netaos::core::Result<void> UpdateAssociatedData (std::uint8_t in)noexcept=0;

private:
    AuthCipherCtx(const AuthCipherCtx&) = delete;
    AuthCipherCtx& operator=(const AuthCipherCtx&) = delete;
    AuthCipherCtx(const AuthCipherCtx&&) = delete;
    AuthCipherCtx& operator=(const AuthCipherCtx&&) = delete;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_AUTH_CIPHER_CTX_H_