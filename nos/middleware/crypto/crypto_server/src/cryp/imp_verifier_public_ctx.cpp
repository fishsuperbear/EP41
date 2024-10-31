#include "cryp/imp_verifier_public_ctx.h"

#include <iostream>
#include <cstddef>
#include <openssl/pem.h>
#include "common/crypto_logger.hpp"
#include "common/imp_volatile_trusted_container.h"
#include "cryp/imp_crypto_provider.h"
#include "cryp/cryobj/imp_public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

netaos::core::Result<void> ImpVerifierPublicCtx::Reset() noexcept {
    return netaos::core::Result<void>();
}

netaos::core::Result<void> ImpVerifierPublicCtx::SetKey(const PublicKey& key) noexcept {
    //上下文初始化
    std::string hash_name;
    // openssl_ctx_.plib_ctx = OSSL_LIB_CTX_new();
    // if (!openssl_ctx_.plib_ctx) {
    //     CRYP_ERROR<<"OSSL_LIB_CTX_new() returned NULL.";
    // }

    switch (alg_id_)
    {
    case kAlgIdRSA2048SHA384PSS:
        hash_name = "SHA2-384";
        openssl_ctx_.pmd = const_cast<EVP_MD*>(EVP_sha384());
        break;
    case kAlgIdRSA2048SHA512PSS:
        hash_name = "SHA2-512";
        openssl_ctx_.pmd = const_cast<EVP_MD*>(EVP_sha512());
        break;
    case kAlgIdRSA2048SHA256PSS:
        hash_name = "SHA2-256";
        openssl_ctx_.pmd = const_cast<EVP_MD*>(EVP_sha256());
        break;
    default:
        break;
    }

    // openssl_ctx_.pmd = EVP_MD_fetch(openssl_ctx_.plib_ctx,static_cast<const char *>(hash_name.data()), NULL);
    if (!openssl_ctx_.pmd) {
        CRYP_ERROR<< "EVP_MD_fetch could not find hash name.";
    }

    openssl_ctx_.pmd_ctx = EVP_MD_CTX_new();
    if (!openssl_ctx_.pmd_ctx) {
        CRYP_ERROR<<"EVP_MD_CTX_new failed.";
    }

    if(EVP_VerifyInit(openssl_ctx_.pmd_ctx, openssl_ctx_.pmd)) {
        ppublic_key_ =const_cast<ImpPublicKey*>(dynamic_cast<const ImpPublicKey*>(&key)); //ok
        // ppublic_key_ = const_cast<PublicKey *>(&key);  //for PublicKey
        isInitialized_ = true;

        CRYP_INFO << "Set ppublic_key_: " << ppublic_key_ << ", evp_pkey: " << ppublic_key_->get_pkey();
    }else{
        isInitialized_ = false;
        CRYP_ERROR<<"EVP_SignInit failed.";
    }

    return netaos::core::Result<void>();

}

netaos::core::Result<bool> ImpVerifierPublicCtx::Verify(ReadOnlyMemRegion value, ReadOnlyMemRegion signature, ReadOnlyMemRegion context) const noexcept{
    bool ret = false;
    //Determine the length of the fetched digest type 
    // openssl_ctx_.verify_len = EVP_MD_get_size(openssl_ctx_.pmd);
    // if (openssl_ctx_.verify_len <= 0) {
    //     CRYP_ERROR<<"EVP_MD_get_size returned invalid size.";
    // }else{
    //     CRYP_ERROR<<"EVP_MD_get_size: "<<openssl_ctx_.verify_len;
    // }

    // openssl_ctx_.pverify_value= (unsigned char*)OPENSSL_malloc(openssl_ctx_.verify_len);
    // if (!openssl_ctx_.pverify_value) {
    //     CRYP_ERROR<< "No memory: pverify_value.";
    // }

    if( EVP_VerifyUpdate(openssl_ctx_.pmd_ctx, value.data(), value.size()) ) {
        CRYP_INFO<< "EVP_VerifyUpdate success.";
    }else{
        CRYP_ERROR<< "EVP_VerifyUpdate failed.";
    }

    CRYP_INFO<< "ppublic_key_ addr."<< ppublic_key_ ;

    // ImpPublicKey *imp_public_key =dynamic_cast<ImpPublicKey*>(ppublic_key_); //ok
    // CRYP_INFO<< "imp_public_key.";
    EVP_PKEY * ppub = ppublic_key_->get_pkey();
    CRYP_INFO << "EVP_PKEY public addr."<< ppub;
    const_cast<ImpVerifierPublicCtx*>(this)->dump_key(ppub);

    CRYP_INFO << "imp_public_key->get_pkey() success.";

    CRYP_INFO << "signature size_bytes:"<< signature.size_bytes();
    CRYP_INFO << "signature size:"<< signature.size();

    CRYP_INFO << "Signature(" << signature.size() << "):\n" << hozon::netaos::crypto::CryptoLogger::GetInstance().ToHexString(signature.data(), signature.size());

    if(1 == EVP_VerifyFinal(openssl_ctx_.pmd_ctx, signature.data(), signature.size(), ppub)){
        CRYP_INFO << "message verify success." ;
        ret = true;
    }else{
        ret = false;
        CRYP_ERROR << "EVP_VerifyFinal() failed. ";
    }

    // if(1 == EVP_VerifyFinal_ex(openssl_ctx_.pmd_ctx, signature.data(), signature.size(), ppub, openssl_ctx_.plib_ctx, nullptr)){
    //     CRYP_INFO << "message verify success." ;
    //     ret = true;
    // }else{
    //     ret = false;
    //     CRYP_ERROR << "EVP_VerifyFinal() failed. ";
    // }
    return netaos::core::Result<bool>::FromValue(ret);
}

CryptoPrimitiveId::Uptr ImpVerifierPublicCtx::GetCryptoPrimitiveId () const noexcept {
    return CryptoPrimitiveId::Uptr(new CryptoPrimitiveId(alg_id_));
}

bool ImpVerifierPublicCtx::IsInitialized() const noexcept{
    return isInitialized_;
}

// CryptoPrimitiveId::Uptr ImpSignerPrivateCtx::GetCryptoPrimitiveId() const noexcept{
//     auto uptr = std::make_unique<CryptoPrimitiveId>();
//     return uptr;
// }

int ImpVerifierPublicCtx::dump_key(const EVP_PKEY *pkey)
{
    #if 0 //openssl 3.0
    int ret = 0;
    int bits = 0;
    BIGNUM *n = NULL, *e = NULL, *d = NULL, *p = NULL, *q = NULL;

   // Retrieve value of n. This value is not secret and forms part of the public key.
    if (EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_N, &n) == 0) {
        CRYP_ERROR<<"Failed to retrieve n";
    }

    // Retrieve value of e. This value is not secret and forms part of the public key. It is typically 65537 and need not be changed.
    if (EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_E, &e) == 0) {
        CRYP_ERROR<<"Failed to retrieve e";
    }

    // Retrieve value of d. This value is secret and forms part of the private key. It must not be published.
    if (EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_D, &d) == 0) {
        CRYP_ERROR<<"Failed to retrieve d";
    }

    // Retrieve value of the first prime factor, commonly known as p. This value is secret and forms part of the private key. It must not be published.
    if (EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_FACTOR1, &p) == 0) {
        CRYP_ERROR<<"Failed to retrieve p";
    }

    // Retrieve value of the second prime factor, commonly known as q. This value is secret and forms part of the private key. It must not be published.
    if (EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_FACTOR2, &q) == 0) {
        CRYP_ERROR<<"Failed to retrieve q";
    }

    // // We can also retrieve the key size in bits for informational purposes.
    // if (EVP_PKEY_get_int_param(pkey, OSSL_PKEY_PARAM_BITS, &bits) == 0) {
    //     CRYP_ERROR<<"Failed to retrieve bits";
    // }

    /* Output hexadecimal representations of the BIGNUM objects. */
    CRYP_INFO<<"Number of bits"<<bits;

    CRYP_INFO<< "Public values:";
    // CRYP_INFO<<"  n = "<<n;
    // CRYP_INFO<<"  e = "<<e;

    CRYP_INFO<< "Private values:";
    // CRYP_INFO<< "  d = "<<d;
    // CRYP_INFO<< "  p = "<<p;
    // CRYP_INFO<< "  q = "<<q;

    PEM_write_PUBKEY(stdout, pkey);
    PEM_write_PrivateKey(stdout, pkey, NULL, NULL, 0, NULL, NULL);

    FILE *prifile = fopen("privatekey.pem","w+");
    if (PEM_write_PrivateKey(prifile, pkey, NULL, NULL, 0, NULL, NULL) == 0) {
        CRYP_ERROR<< "Failed to output PEM-encoded private key";
    }
    fclose(prifile);

    FILE *pubfile = fopen("publickey.pem","w+");
    if (PEM_write_PUBKEY(pubfile, pkey) == 0) {
        CRYP_ERROR<< "Failed to output PEM-encoded private key";
    }
    fclose(pubfile);

    // char filename[] = "privatekeyB.txt";
    // BIO* bp = BIO_new(BIO_s_file());
    // BIO_write_filename(bp,filename);
    // if (!PEM_write_bio_PrivateKey(bp, pkey, NULL, NULL, 0, 0, NULL)){
    // }
    // BIO_flush(bp);
    // BIO_free(bp);
   
    BN_free(n); /* not secret */
    BN_free(e); /* not secret */
    BN_clear_free(d); /* secret - scrub before freeing */
    BN_clear_free(p); /* secret - scrub before freeing */
    BN_clear_free(q); /* secret - scrub before freeing */
    return ret;
    #endif
    return 0;
}


CryptoProvider& ImpVerifierPublicCtx::MyProvider() const noexcept{
    // return const_cast<ImpCryptoProvider&>(ImpCryptoProvider::Instance());
    // CryptoProvider& prov = ImpCryptoProvider::Instance();
    CryptoProvider* prov = new ImpCryptoProvider;
    return *prov;
}

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
