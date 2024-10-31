#ifndef ARA_CRYPTO_CRYP_SECRET_SEED_H_
#define ARA_CRYPTO_CRYP_SECRET_SEED_H_
#include "core/result.h"
#include "common/mem_region.h"
#include "common/base_id_types.h"
#include "cryp/cryobj/restricted_use_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class SecretSeed:public RestrictedUseObject{
public:

    using Uptrc = std::unique_ptr<const SecretSeed>;
    using Uptr = std::unique_ptr<SecretSeed>;
    virtual netaos::core::Result<SecretSeed::Uptr> Clone(ReadOnlyMemRegion xorDelta = ReadOnlyMemRegion()) const noexcept = 0;
    virtual netaos::core::Result<void> JumpFrom(const SecretSeed& from, std::int64_t steps) noexcept = 0;
    virtual SecretSeed& Jump(std::int64_t steps) noexcept = 0;
    virtual SecretSeed& Next() noexcept = 0;
    virtual SecretSeed& operator=(const SecretSeed& source) noexcept = 0;
    virtual SecretSeed& operator=(ReadOnlyMemRegion source) noexcept = 0;
    static const CryptoObjectType kObjectType = CryptoObjectType::kSecretSeed;

   private:
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_SECRET_SEED_H_