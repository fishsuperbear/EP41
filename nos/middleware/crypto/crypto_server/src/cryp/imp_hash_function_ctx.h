#ifndef ARA_CRYPTO_CRYP_IMP_HASH_FUNCTION_CTX_H_
#define ARA_CRYPTO_CRYP_IMP_HASH_FUNCTION_CTX_H_

#include <vector>
#include <utility>
#include <string.h>
#include <stdio.h>

#include "cryp/hash_function_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "openssl/err.h"
#include "openssl/evp.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
    
class ImpHashFunctionCtx: public HashFunctionCtx{
public:
    struct OpensslCtx
    {
        OSSL_LIB_CTX* plibrary_context;
        EVP_MD* pmessage_digest;
        EVP_MD_CTX* pdigest_context;
        const char* poption_properties;
        unsigned char* pdigest_value;
        unsigned int digest_length;
        OpensslCtx():plibrary_context(NULL),pmessage_digest(NULL),pdigest_context(NULL),poption_properties(NULL),pdigest_value(),digest_length(0){}
    };
    
    // ImpHashFunctionCtx();
    ImpHashFunctionCtx(AlgId id):alg_id_(id){}

    using Uptr = std::unique_ptr<ImpHashFunctionCtx>;
    netaos::core::Result<netaos::core::Vector<uint8_t>>  Finish() noexcept override;
    // DigestService::Uptr GetDigestService() const noexcept override; //TODO
    netaos::core::Result<std::vector<uint8_t>> GetDigest(std::size_t offset = 0) const noexcept override;
    // template <typename Alloc = <implementation-defined>>
    // template <typename Alloc = uint8_t>
    // ByteVector<Alloc> GetDigest(std::size_t offset = 0) const noexcept;
    netaos::core::Result<void> Start() noexcept override;
    // void Start(ReadOnlyMemRegion iv) noexcept override;
    // void Start(const SecretSeed& iv) noexcept = 0;
    // void Update(const RestrictedUseObject& in) noexcept override;
    // void Update(ReadOnlyMemRegion& in) noexcept override;
    netaos::core::Result<void> Update(std::vector<uint8_t>& in) noexcept override;
    // void Update(std::uint8_t in) noexcept override;
    CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
    bool IsInitialized () const noexcept override;
    CryptoProvider& MyProvider() const noexcept override;

    // void FreeOpensslCtx() noexcept;

    
    // using Uptr = std::unique_ptr<HashFunctionCtx>;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > Finish() noexcept = 0;
    // virtual DigestService::Uptr GetDigestService() const noexcept = 0;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > GetDigest(std::size_t offset = 0) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> GetDigest(std::size_t offset = 0) const noexcept;
    // virtual ara::core::Result<void> Start() noexcept = 0;
    // virtual ara::core::Result<void> Start(ReadOnlyMemRegion iv) noexcept = 0;
    // virtual ara::core::Result<void> Start(const SecretSeed& iv) noexcept = 0;
    // virtual ara::core::Result<void> Update(const RestrictedUseObject& in) noexcept = 0;
    // virtual ara::core::Result<void> Update(ReadOnlyMemRegion in) noexcept = 0;
    // virtual ara::core::Result<void> Update(std::uint8_t in) noexcept = 0;

private:
    CryptoAlgId alg_id_ = kAlgIdUndefined;
    bool isInitialized_ = false;

    std::vector<std::uint8_t> input_;
    OpensslCtx openssl_ctx_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#ifdef __cplusplus
}
#endif

#endif  // #define ARA_CRYPTO_CRYP_IMP_HASH_FUNCTION_CTX_H_