#ifndef ARA_CRYPTO_IMP_CRYP_PRIVATE_KEY_H_
#define ARA_CRYPTO_IMP_CRYP_PRIVATE_KEY_H_

#include <openssl/rsa.h>
#include <openssl/err.h>
#include <openssl/evp.h>
// #include <openssl/core_names.h>
#include <openssl/pem.h>
// #include <openssl/types.h>
#include "core/result.h"
#include "cryp/cryobj/private_key.h"
#include "cryp/cryobj/public_key.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {
class ImpPrivateKey : public PrivateKey{
public:
    using Uptrc = std::unique_ptr<const ImpPrivateKey>;
    using AlgId = CryptoPrimitiveId::AlgId;
    static const CryptoObjectType kObjectType = CryptoObjectType::kPrivateKey;

    ImpPrivateKey(CryptoObjectInfo& object_info, CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage)
    : PrivateKey(object_info, primitive_id, usage) {
        EVP_PKEY_CTX *genctx = NULL;
        OSSL_LIB_CTX *libctx = NULL;
        const char *propq = NULL;
        std::string name ;

        if(crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA2048SHA256PSS || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA2048SHA512PSS
            || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA2048SHA384PKCS || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA2048SHA512PKCS){
            keyBitLenth_ = 2048;
            name = "RSA";
        }else if(crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA3072SHA256PSS || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA3072SHA512PSS
            || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA3072SHA256PKCS || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA3072SHA512PKCS){
            keyBitLenth_ = 3072;
            name = "RSA";
        }else if(crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA4096SHA256PSS || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA4096SHA384PSS
            || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA4096SHA256PKCS || crypto_primitive_id_.GetPrimitiveId() == kAlgIdRSA4096SHA384PKCS){
            keyBitLenth_ = 4096;
            name = "RSA";
        }else{

        }

        // Create context using RSA algorithm.
        genctx = EVP_PKEY_CTX_new_from_name(libctx, static_cast<const char *>(name.data()), propq);
        if (genctx == NULL) {
            OSSL_LIB_CTX_free(libctx);
            CRYP_ERROR << "EVP_PKEY_CTX_new_from_name fail.";
        }
        CRYP_INFO << "EVP_PKEY_CTX_new_from_name finish.";

        //Initialize context for key generation purposes. 
        if (EVP_PKEY_keygen_init(genctx) <= 0) {
            EVP_PKEY_CTX_free(genctx);
            CRYP_ERROR << "EVP_PKEY_keygen_init fail.";

        }
        CRYP_INFO << "EVP_PKEY_keygen_init finish.";

        if (EVP_PKEY_CTX_set_rsa_keygen_bits(genctx, keyBitLenth_) <= 0) {
            EVP_PKEY_CTX_free(genctx);
            CRYP_ERROR << "EVP_PKEY_CTX_set_rsa_keygen_bits fail.";
        }
        CRYP_INFO << "EVP_PKEY_CTX_set_rsa_keygen_bits finish.";

        if (EVP_PKEY_generate(genctx, &pkey_) <= 0) {
            EVP_PKEY_CTX_free(genctx);
            CRYP_ERROR << "EVP_PKEY_generate fail.";
        }
        CRYP_INFO << "EVP_PKEY_generate finish.";

        // dump_key(pkey_);
    }

    ImpPrivateKey(EVP_PKEY *pkey, CryptoObjectInfo& object_info, CryptoPrimitiveId& primitive_id, AllowedUsageFlags& usage)
    : PrivateKey(object_info, primitive_id, usage) {
        pkey_ = EVP_PKEY_dup(pkey);
    }

    ~ImpPrivateKey();

    netaos::core::Result<PublicKey::Uptrc> GetPublicKey() const noexcept override;
    bool CheckKey(bool strongCheck = true) const noexcept override;
    
    // Usage GetAllowedUsage() const noexcept override;
    // CryptoPrimitiveId::Uptr GetCryptoPrimitiveId () const noexcept override;
    // COIdentifier GetObjectId () const noexcept override;
    // COIdentifier HasDependence () const noexcept override;
    // std::size_t GetPayloadSize () const noexcept override;
    // bool IsExportable () const noexcept override;
    // bool IsSession () const noexcept override;
    netaos::core::Result<void> Save(IOInterface& container) const noexcept override;

    int dump_key(const EVP_PKEY* pkey);//TODO for test
    EVP_PKEY * get_pkey(){
        return pkey_;
    };

private:

    // netaos::core::StringView primitiveName_;
    EVP_PKEY *pkey_ = NULL; // private-public pair
    int keyBitLenth_;
};
}  // namespace cryp
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon

#endif  // #define ARA_CRYPTO_CRYP_PRIVATE_KEY_H_