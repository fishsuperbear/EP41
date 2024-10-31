#ifndef ARA_CRYPTO_CRYP_SIGNATURE_H_
#define ARA_CRYPTO_CRYP_SIGNATURE_H_

#include "crypto_object.h"
#include "crypto_primitive_id.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class Signature:public CryptoObject{
public:
    using Uptrc = std::unique_ptr<const Signature>;
    // Signature(bool isSession, bool isExportable):CryptoObject(isSession,isExportable){}
    Signature(const CryptoObjectInfo& object_info, const CryptoPrimitiveId& primitive_id)
    : CryptoObject(object_info, primitive_id) {

    }
    virtual CryptoPrimitiveId::AlgId GetHashAlgId () const noexcept {
        CryptoAlgId hash_alg_id = kAlgIdUndefined;
        switch (crypto_primitive_id_.GetPrimitiveId()) {
            case kAlgIdRSA2048SHA384PSS:
            hash_alg_id = kAlgIdSHA384;
            break;
            case kAlgIdRSA2048SHA512PSS:
            hash_alg_id = kAlgIdSHA512;
            break;
            case kAlgIdRSA3072SHA256PSS:
            hash_alg_id = kAlgIdSHA256;
            break;
            case kAlgIdRSA3072SHA512PSS:
            hash_alg_id = kAlgIdSHA512;
            break;
            case kAlgIdRSA4096SHA256PSS:
            hash_alg_id = kAlgIdSHA256;
            break;
            case kAlgIdRSA4096SHA384PSS:
            hash_alg_id = kAlgIdSHA384;
            break;
            case kAlgIdRSA2048SHA384PKCS:
            hash_alg_id = kAlgIdSHA384;
            break;
            case kAlgIdRSA2048SHA512PKCS:
            hash_alg_id = kAlgIdSHA512;
            break;
            case kAlgIdRSA3072SHA256PKCS:
            hash_alg_id = kAlgIdSHA256;
            break;
            case kAlgIdRSA3072SHA512PKCS:
            hash_alg_id = kAlgIdSHA512;
            break;
            case kAlgIdRSA4096SHA256PKCS:
            hash_alg_id = kAlgIdSHA256;
            break;
            case kAlgIdRSA4096SHA384PKCS:
            hash_alg_id = kAlgIdSHA384;
            break;
            default:
            break;
        }

        return hash_alg_id;
    }

    virtual std::size_t GetRequiredHashSize() const noexcept {
        std::size_t hash_size = 0;
        switch (GetHashAlgId()) {
            case kAlgIdMD5:
            hash_size = 16;
            break;
            case kAlgIdSHA1:
            hash_size = 20;
            break;
            case kAlgIdSHA256:
            hash_size = 32;
            break;
            case kAlgIdSHA384:
            hash_size = 48;
            break;
            case kAlgIdSHA512:
            hash_size = 64;
            break;
            default:
            break;
        }

        return hash_size;
    }
    static const CryptoObjectType kObjectType = CryptoObjectType::kSignature;
private:
 
};
}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara

#endif  // #define ARA_CRYPTO_CRYP_SIGNATURE_H_