#include "cryp/imp_decryptor_private_ctx.h"

#include <memory>
#include <openssl/err.h>
#include <openssl/evp.h>
// #include <openssl/types.h>
#include <openssl/rsa.h>

#include "cryp/imp_crypto_provider.h"
#include "cryp/imp_crypto_service.h"
#include "cryp/cryobj/imp_private_key.h"
#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

static const unsigned char aes_key[] = "0123456789abcdeF";

CryptoService::Uptr ImpDecryptorPrivateCtx::GetCryptoService() const noexcept{
    return std::make_unique<ImpCryptoService>();
}

netaos::core::Result<netaos::core::Vector<uint8_t>> ImpDecryptorPrivateCtx::ProcessBlock(ReadOnlyMemRegion in, bool suppressPadding) const noexcept {
    size_t inlen = in.size();
    const unsigned char *indata = in.data();
    size_t outlen = 0;
    unsigned char *outdata = NULL;
    
    if(suppressPadding){
        if (EVP_PKEY_CTX_set_rsa_padding(pkey_ctx_, RSA_PKCS1_OAEP_PADDING) <= 0){
            CRYP_ERROR << "EVP_PKEY_CTX_set_rsa_padding error.";
        }
    }

    if (EVP_PKEY_decrypt(pkey_ctx_, NULL, &outlen, indata, inlen) <= 0) {
        CRYP_ERROR << "EVP_PKEY_decrypt error.";
    }

    outdata = static_cast<unsigned char*>(OPENSSL_malloc(outlen));

    if (!outdata) {
        CRYP_ERROR << "malloc failure. error.";
    }

    if (EVP_PKEY_decrypt(pkey_ctx_, outdata, &outlen, indata, inlen) <= 0) {
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

netaos::core::Result<void> ImpDecryptorPrivateCtx::Reset() noexcept{
    return netaos::core::Result<void>();
}

netaos::core::Result<void> ImpDecryptorPrivateCtx::SetKey(const PrivateKey& key) noexcept{
    ImpPrivateKey& imp_private_key = dynamic_cast<ImpPrivateKey&>(const_cast<PrivateKey&>(key));
    // EVP_PKEY* pkey = EVP_PKEY_new();
    EVP_PKEY *pkey = imp_private_key.get_pkey();
    pkey_ctx_ = EVP_PKEY_CTX_new(pkey, NULL);
    if (!pkey_ctx_){
        CRYP_ERROR<< "EVP_PKEY_CTX_new error.";
    }
    
    if (EVP_PKEY_decrypt_init(pkey_ctx_) <= 0){
        CRYP_ERROR<< "EVP_PKEY_decrypt_init error.";
    }
    return netaos::core::Result<void>();
}

CryptoPrimitiveId::Uptr ImpDecryptorPrivateCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

bool ImpDecryptorPrivateCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

CryptoProvider& ImpDecryptorPrivateCtx::MyProvider() const noexcept{
    CryptoProvider* prov = new ImpCryptoProvider;
    return *prov;
}

// ImpSymmetricBlockCipherCtx::ImpSymmetricBlockCipherCtx(){

// }

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
