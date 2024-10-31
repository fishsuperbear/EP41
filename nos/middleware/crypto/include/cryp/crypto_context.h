#ifndef ARA_CRYPTO_CRYP_CRYPTO_CONTEXT_H_
#define ARA_CRYPTO_CRYP_CRYPTO_CONTEXT_H_

#include "cryp/cryobj/crypto_primitive_id.h"
// #include "crypto_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class CryptoProvider;

class CryptoContext {
public:
    using AlgId = CryptoAlgId;
    virtual ~CryptoContext () noexcept=default;
    virtual CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept=0;
    virtual bool IsInitialized () const noexcept=0;
    CryptoContext& operator= (const CryptoContext &other)=default;
    CryptoContext& operator= (CryptoContext &&other)=default;
    // CryptoContext(const CryptoContext&) = default;
    // CryptoContext(const CryptoContext&&) = default;
    virtual CryptoProvider& MyProvider() const noexcept = 0;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_CRYPTO_CONTEXT_H_