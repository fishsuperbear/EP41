#include "cryp/imp_encryptor_public_ctx.h"

#include <memory>

#include "openssl/err.h"
#include "openssl/evp.h"
// #include "openssl/types.h"
#include "openssl/rsa.h"

#include "cryp/imp_crypto_provider.h"
#include "cryp/imp_crypto_service.h"
#include "cryp/cryobj/imp_public_key.h"
#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

static const unsigned char aes_key[] = "0123456789abcdeF";

CryptoService::Uptr ImpEncryptorPublicCtx::GetCryptoService() const noexcept{
    return std::make_unique<ImpCryptoService>();
}

netaos::core::Result<netaos::core::Vector<uint8_t>> ImpEncryptorPublicCtx::ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding) const noexcept {
    size_t inlen = in.size();
    const unsigned char *indata = in.data();
    size_t outlen = 0;
    unsigned char *outdata = NULL;
    
    if(suppressPadding){
        if (EVP_PKEY_CTX_set_rsa_padding(pkey_ctx_, RSA_PKCS1_OAEP_PADDING) <= 0){
            CRYP_ERROR << "EVP_PKEY_CTX_set_rsa_padding error.";
        }
    }

    if (EVP_PKEY_encrypt(pkey_ctx_, NULL, &outlen, indata, inlen) <= 0) {
        CRYP_ERROR << "EVP_PKEY_encrypt error.";
    }
    CRYP_INFO << "EVP_PKEY_decrypt outlen:" <<outlen;

    outdata = static_cast<unsigned char*>(OPENSSL_malloc(outlen));

    if (!outdata) {
        CRYP_ERROR << "malloc failure. error.";
    }

    if (EVP_PKEY_encrypt(pkey_ctx_, outdata, &outlen, indata, inlen) <= 0) {
        CRYP_ERROR << "EVP_PKEY_decrypt error.";
    }

    CRYP_INFO << "outdata:" << outdata;
    // ara::core::Vector<uint8_t> out(static_cast<uint8_t*>(outdata),outlen);
    netaos::core::Vector<uint8_t> out;
    out.resize(outlen);
    for(size_t i = 0;i<outlen;i++){
        out[i] = static_cast<uint8_t>(outdata[i]);
    }
    return netaos::core::Result<netaos::core::Vector<uint8_t>>::FromValue(std::move(out));

    // if(inlength > 1024){
    //     while (inlength > 1024) {
    //         if (!EVP_CipherUpdate(ctx_, poutdata, &outtemplen, pindata, inlength)) {
    //             CRYP_ERROR << "EVP_CipherUpdate error.";
    //         }
    //         pindata += 1024;
    //         inlength -= 1024;
    //         poutdata += outtemplen;
    //         outlen_ += outtemplen;
    //     }
    // }else{
    //     if (!EVP_CipherUpdate(ctx_, poutdata, &outtemplen, pindata, inlength)) {
    //         CRYP_ERROR << "EVP_CipherUpdate error.";
    //     }
    //     poutdata += outtemplen;
    //     outlen_ += outtemplen;
    // }

}

netaos::core::Result<void> ImpEncryptorPublicCtx::Reset() noexcept{
    return netaos::core::Result<void>();
}

netaos::core::Result<void> ImpEncryptorPublicCtx::SetKey(const PublicKey& key) noexcept{

    CRYP_INFO<< "ImpEncryptorPublicCtx SetKey start.";
    // ImpPublicKey& imp_public_key = dynamic_cast<ImpPublicKey&>(const_cast<PublicKey&>(key));
    // ImpPublicKey& imp_public_key = const_cast<ImpPublicKey&> (dynamic_cast<const ImpPublicKey&>(key));
    // try {
    // CRYP_INFO << "key name:" << key.get_myName();
    // CRYP_INFO << "key name:" << key.get_myName();

    CRYP_INFO << "key addr:" << &key;
    // ImpPublicKey *imp_public_key = nullptr;


    // ImpPublicKey& imp_public_key = dynamic_cast<ImpPublicKey&>(key);
    // ImpPublicKey *imp_public_key =(dynamic_cast<ImpPublicKey*>(const_cast<PublicKey*>(&key))); 


    // ImpPublicKey imp_public_key() 
    // CRYP_INFO << "key name:" << imp_public_key.get_myName();

    // imp_public_key =const_cast<ImpPublicKey*>(dynamic_cast<const ImpPublicKey*>(&key)); 
    // const ImpPublicKey * imp_public_key = dynamic_cast<const ImpPublicKey*>(&key); 
    ImpPublicKey *imp_public_key =const_cast<ImpPublicKey*>(dynamic_cast<const ImpPublicKey*>(&key)); //ok
    // ImpPublicKey& imp_public_key = const_cast<ImpPublicKey&> (dynamic_cast<const ImpPublicKey&>(key));

    if(imp_public_key != NULL){
        CRYP_INFO << "ImpEncryptorPublicCtx dynamic_cast over.";
    }

    // EVP_PKEY* pkey = (dynamic_cast<const ImpPublicKey*>(&key)).get_pkey();
    EVP_PKEY* pkey = imp_public_key->get_pkey();
    pkey_ctx_ = EVP_PKEY_CTX_new(pkey, NULL);
    if (pkey_ctx_) {
        if (EVP_PKEY_encrypt_init(pkey_ctx_) <= 0) {
            CRYP_ERROR << "EVP_PKEY_decrypt_init error.";
        }
    }
    return netaos::core::Result<void>();
}

CryptoPrimitiveId::Uptr ImpEncryptorPublicCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

bool ImpEncryptorPublicCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

CryptoProvider& ImpEncryptorPublicCtx::MyProvider() const noexcept{
    CryptoProvider* prov = new ImpCryptoProvider;
    return *prov;
}

// ImpSymmetricBlockCipherCtx::ImpSymmetricBlockCipherCtx(){

// }

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
