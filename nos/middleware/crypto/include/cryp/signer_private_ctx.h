#ifndef ARA_CRYPTO_CRYP_SIGNER_RRIVATE_CTX_H_
#define ARA_CRYPTO_CRYP_SIGNER_RRIVATE_CTX_H_
#include "core/result.h"
#include "common/mem_region.h"
#include "cryp/crypto_context.h"
#include "cryp/signature_service.h"
#include "cryp/cryobj/signature.h"
#include "cryp/cryobj/private_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class SignerPrivateCtx:public CryptoContext{
public:

    const PrivateKey::Uptrc uptrc_privateKey;

    using Uptr = std::unique_ptr<SignerPrivateCtx>;

    // virtual SignatureService::Uptr GetSignatureService() const noexcept = 0;
    virtual netaos::core::Result<void> Reset() noexcept = 0;
    virtual netaos::core::Result<void> SetKey(const PrivateKey& key) noexcept = 0;
    // virtual ara::core::Result<Signature::Uptrc> SignPreHashed (const HashFunctionCtx &hashFn, ReadOnlyMemRegion context=ReadOnlyMemRegion()) const noexcept=0;
    // virtual ara::core::Result<Signature::Uptrc> SignPreHashed(AlgId hashAlgId, ReadOnlyMemRegion hashValue, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>>Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> Sign(ReadOnlyMemRegion value, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept;

    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> SignPreHashed (const HashFunctionCtx &hashFn, ReadOnlyMemRegion context=ReadOnlyMemRegion()) const noexcept=0;  // add by neta
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> SignPreHashed(AlgId hashAlgId, ReadOnlyMemRegion hashValue, ReadOnlyMemRegion context = ReadOnlyMemRegion()) const noexcept = 0;// add by neta

    virtual netaos::core::Result<PublicKey::Uptrc> GetPublicKey() const noexcept = 0; // add by neta
   private:
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_SIGNER_RRIVATE_CTX_H_