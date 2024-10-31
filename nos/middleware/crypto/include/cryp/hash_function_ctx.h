#ifndef ARA_CRYPTO_CRYP_HASH_FUNCTION_CTX_H_
#define ARA_CRYPTO_CRYP_HASH_FUNCTION_CTX_H_

#include <vector>
#include "core/result.h"
#include "core/vector.h"
#include "common/mem_region.h"
#include "cryp/crypto_context.h"
#include "cryp/cryobj/secret_seed.h"
#include "cryp/digest_service.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

class HashFunctionCtx:public CryptoContext{
public:
    using Uptr = std::unique_ptr<HashFunctionCtx>;
    // virtual ara::core::Result<ara::core::Vector<ara::core::Byte> > Finish() noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t>> Finish() noexcept = 0;
    // virtual DigestService::Uptr GetDigestService() const noexcept = 0;
    virtual netaos::core::Result<netaos::core::Vector<uint8_t> > GetDigest(std::size_t offset = 0) const noexcept = 0;
    // template <typename Alloc = <implementation - defined>>
    // ara::core::Result<ByteVector<Alloc>> GetDigest(std::size_t offset = 0) const noexcept;
    virtual netaos::core::Result<void> Start() noexcept = 0;
    // virtual ara::core::Result<void> Start(ReadOnlyMemRegion iv) noexcept = 0;
    // virtual ara::core::Result<void> Start(const SecretSeed& iv) noexcept = 0;
    // virtual ara::core::Result<void> Update(const RestrictedUseObject& in) noexcept = 0;
    // virtual ara::core::Result<void> Update(ReadOnlyMemRegion in) noexcept = 0;
    virtual netaos::core::Result<void> Update(std::vector<uint8_t>& in) noexcept = 0;  //TODO  for test.need to delete

    // virtual ara::core::Result<void> Update(std::uint8_t in) noexcept = 0;

private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_HASH_FUNCTION_CTX_H_