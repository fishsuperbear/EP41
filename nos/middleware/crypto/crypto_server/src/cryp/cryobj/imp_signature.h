#ifndef ARA_CRYPTO_CRYP_IMP_SIGNATURE_H_
#define ARA_CRYPTO_CRYP_IMP_SIGNATURE_H_
#include "crypto_object.h"
#include "crypto_primitive_id.h"
#include "signature.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpSignature:public Signature{
public:
    using Uptrc = std::unique_ptr<const ImpSignature>;
    using AlgId = CryptoPrimitiveId::AlgId;
    ImpSignature(AlgId hashAlgid, AlgId signAlgid, bool isSession, bool isExportable): \
    Signature(isSession,isExportable),hashAlgid_(hashAlgid),signAlgId_(signAlgid){
        switch (hashAlgid)
        {
        case kAlgIdSHA256:
            hashSize_ = 256/8;
            break;
        case kAlgIdSHA384:
            hashSize_ = 384/8;
            break;
        case kAlgIdSHA512:
            hashSize_ = 512/8;
            break;
        default:
            break;
        }
    }
    CryptoPrimitiveId::AlgId GetHashAlgId () const noexcept override;
    std::size_t GetRequiredHashSize() const noexcept override;
private:
    AlgId hashAlgId_;
    AlgId signAlgId_; 
    std::size_t hashSize_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_IMP_SIGNATURE_H_