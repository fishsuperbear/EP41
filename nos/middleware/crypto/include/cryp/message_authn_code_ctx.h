#ifndef ARA_CRYPTO_CRYP_MESSAGE_AUTHN_CODE_CTX_H_
#define ARA_CRYPTO_CRYP_MESSAGE_AUTHN_CODE_CTX_H_

#include "core/result.h"
#include "cryp/cryobj/crypto_object.h"
#include "cryp/crypto_context.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class MessageAuthnCodeCtx:public CryptoContext{
public:

    using Uptr = std::unique_ptr<MessageAuthnCodeCtx>;
    virtual netaos::core::Result<bool> Check(const Signature& expected) const noexcept = 0;
    virtual netaos::core::Result<Signature::Uptrc> Finish(bool makeSignatureObject = false) noexcept = 0;
    virtual DigestService::Uptr GetDigestService () const noexcept=0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > GetDigest(std::size_t offset = 0) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t> > GetDigest(std::size_t offset = 0) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> GetDigest(std::size_t offset = 0) const noexcept;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const SymmetricKey& key, CryptoTransform transform = CryptoTransform::kMacGenerate) noexcept = 0;
    virtual netaos::core::Result<void> Start(ReadOnlyMemRegion iv = ReadOnlyMemRegion()) noexcept = 0;
    virtual netaos::core::Result<void> Start(const SecretSeed& iv) noexcept = 0;
    virtual netaos::core::Result<void> Update(const RestrictedUseObject& in) noexcept = 0;
    virtual netaos::core::Result<void> Update(ReadOnlyMemRegion in) noexcept = 0;
    virtual netaos::core::Result<void> Update(std::uint8_t in) noexcept = 0;

    private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_MESSAGE_AUTHN_CODE_CTX_H_