#ifndef ARA_CRYPTO_CRYP_SIGNATURE_SERVICE_H_
#define ARA_CRYPTO_CRYP_SIGNATURE_SERVICE_H_

#include "cryp/cryobj/crypto_primitive_id.h"
#include "cryp/cryobj/crypto_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class SignatureService:public CryptoObject{
public:
    virtual CryptoPrimitiveId::AlgId GetRequiredHashAlgId() const noexcept = 0;
    virtual std::size_t GetRequiredHashSize () const noexcept=0;
    virtual std::size_t GetSignatureSize() const noexcept = 0;

   private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_SIGNATURE_SERVICE_H_