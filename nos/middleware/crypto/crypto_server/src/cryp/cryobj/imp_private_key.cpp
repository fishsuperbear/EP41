#include "cryp/cryobj/imp_private_key.h"

#include <memory>
#include <cstdio>
#include "openssl/bio.h"
#include "common/imp_io_interface.h"
#include "common/crypto_logger.hpp"
#include "cryp/cryobj/crypto_object.h"
// #include "cryp/cryobj/imp_crypto_primitive_id.hpp"
#include "cryp/cryobj/imp_public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace cryp {

ImpPrivateKey::~ImpPrivateKey() {
    CRYP_INFO << "ImpPrivateKey destructor. add: 0x" << this;
    if(pkey_){
        EVP_PKEY_free(pkey_);
    }
}

netaos::core::Result<void> ImpPrivateKey::Save(IOInterface& container) const noexcept{
    auto key = const_cast<ImpPrivateKey*>(this)->get_pkey();
    if (!key) {
        CRYP_ERROR << "ImpPrivateKey get_pkey is null";
        return netaos::core::Result<void>();
    }
    char* buffer = nullptr;
    int len = 0;
    std::string privateKeyPEM;
    // 创建BIO对象来存储缓冲区
    BIO* bio = BIO_new(BIO_s_mem());
    if (bio) {
        // 将私钥以PEM编码存储到缓冲区中
        if (PEM_write_bio_PrivateKey(bio, key, nullptr, nullptr, 0, nullptr, nullptr)) {
            // 从BIO对象中读取缓冲区内容到字符串中
            len = BIO_get_mem_data(bio, &buffer);
            privateKeyPEM.assign(buffer, len);
        } else {
            CRYP_ERROR << "Failed to write private key to BIO.";
        }
        // 释放BIO对象
        BIO_free(bio);
    } else {
        CRYP_ERROR << "Failed to create BIO.";
    }
    if (privateKeyPEM.size() == 0) {
        CRYP_ERROR << "privateKeyPEM error";
        return netaos::core::Result<void>();
    }
    std::vector<uint8_t> payload(privateKeyPEM.begin(), privateKeyPEM.end());
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.usage = const_cast<ImpPrivateKey*>(this)->GetAllowedUsage();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectSize = const_cast<ImpPrivateKey*>(this)->GetPayloadSize();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.isSession = const_cast<ImpPrivateKey*>(this)->IsSession();
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectType = const_cast<ImpPrivateKey*>(this)->GetObjectId().mCOType;
    dynamic_cast<ImpIOInterface&>(container).ioInfo_.objectUid = const_cast<ImpPrivateKey*>(this)->GetObjectId().mCouid;

    dynamic_cast<ImpIOInterface&>(container).SetPayload(payload);
    return netaos::core::Result<void>();
};

netaos::core::Result<PublicKey::Uptrc> ImpPrivateKey::GetPublicKey() const noexcept{
    int type = 0;
    unsigned char pub[4096] = {};
    size_t publen = 0;
    EVP_PKEY *pubkey = EVP_PKEY_new();
    PublicKey::Uptrc uptrc = nullptr;
    type = EVP_PKEY_get_base_id(pkey_);
    CRYP_INFO << "EVP_PKEY_get_base_id type:" <<type;
    if(type == EVP_PKEY_RSA){
        EVP_PKEY* pPublicKey = nullptr;
        RSA *rsa = nullptr;
        RSA *rsa_publickey = nullptr;

        pPublicKey = EVP_PKEY_new();
        if (!pPublicKey) {
            CRYP_ERROR << "GetPublicKey pPublicKey is null";
        }
        rsa = EVP_PKEY_get1_RSA(pkey_);
        if (rsa) {
            rsa_publickey = RSAPublicKey_dup(rsa);
        } else {
            CRYP_ERROR << "GetPublicKey rsa is null";
        }

        if (rsa_publickey) {
            if (1 != EVP_PKEY_set1_RSA(pPublicKey, rsa_publickey)) {
                CRYP_ERROR << "Call EVP_PKEY_set1_RSA failed.";
            }
        } else {
            CRYP_ERROR << "GetPublicKey rsa_publickey is null";
        }
        pubkey = pPublicKey;
        // pubkey = pkey_;
        if (pubkey) {
            const_cast<ImpPrivateKey*>(this)->dump_key(pubkey);
        }

    }else if(type == EVP_PKEY_X25519 || type == EVP_PKEY_ED25519 || type == EVP_PKEY_X448 || type == EVP_PKEY_ED448){
        if(EVP_PKEY_get_raw_public_key(pubkey, pub,&publen)){
            pubkey = EVP_PKEY_new_raw_public_key(type, NULL,pub,publen);
        }else{
            CRYP_ERROR << "EVP_PKEY_get_raw_public_key failed." ;
        }
    }else{

    }

    // if(pubkey){
        CryptoObject::CryptoObjectInfo pub_object_info = crypto_object_info_;
        pub_object_info.isExportable = true;
        AllowedUsageFlags usage = kAllowVerification | kAllowDataEncryption;
        CryptoPrimitiveId crypto_primitive_id = crypto_primitive_id_;

        ImpPublicKey* simpl_pub_key = new ImpPublicKey(pubkey, pub_object_info, crypto_primitive_id, usage);
        uptrc.reset(simpl_pub_key);
    // }else{
    //     CRYP_ERROR << "uptrc is null" ;
    // }

    CRYP_INFO << "simpl_pub_key: " << uptrc.get() << ", evp_pkey: " << pubkey;
    return netaos::core::Result<PublicKey::Uptrc>::FromValue(std::move(uptrc));
};

bool ImpPrivateKey::CheckKey(bool strongCheck) const noexcept{
    return true;
};

// ara::core::Result<CryptoTransform> ImpSymmetricBlockCipherCtx::GetTransformation() const noexcept {
//     return ara::core::Result<CryptoTransform>::FromValue(transform_);
// };


int ImpPrivateKey::dump_key(const EVP_PKEY *pkey)
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

    // We can also retrieve the key size in bits for informational purposes.
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

}  // namespace cryp
}  // namespace crypto
}  // namespace ara
}  // namespace ara
