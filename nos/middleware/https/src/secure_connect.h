/*
* Copyright (c) hozonauto. 2021-2021. All rights reserved.
* Description: Http IF class
*/

#ifndef V2C_HTTPLIB_IMPL_SECURE_CONNECT_H
#define V2C_HTTPLIB_IMPL_SECURE_CONNECT_H

// #include <openssl/x509v3.h>
// #include <openssl/x509_vfy.h>
// #include <openssl/crypto.h>
// #include <openssl/lhash.h>
// #include <openssl/objects.h>
// #include <openssl/err.h>
// #include <openssl/evp.h>
// #include <openssl/x509.h>
// #include <openssl/pkcs12.h>
// #include <openssl/bio.h>
#include <ctime>
#include <string>
#include <memory>
#include <vector>

#include "openssl/ossl_typ.h"
#include "openssl/ssl.h"
#include "openssl/bn.h"
#include "openssl/rsa.h"

namespace hozon {
namespace netaos {
namespace https {

#define make_shared_openssl_rsa() std::shared_ptr<RSA>(RSA_new(), [](RSA* p) { RSA_free(p);})
#define make_shared_openssl_bn() std::shared_ptr<BIGNUM>(BN_new(), [](BIGNUM* p) { BN_free(p);})
#define make_shared_openssl_bio(data, size) std::shared_ptr<BIO>(BIO_new_mem_buf(data, size), [](BIO* bio) { BIO_free(bio); })
#define make_shared_openssl_pkey(pkey) std::shared_ptr<EVP_PKEY>(EVP_PKEY_new(), [](EVP_PKEY* p) { EVP_PKEY_free(p); })
#define use_shared_openssl_rsa(rsa) std::shared_ptr<RSA>(rsa, [](RSA* p) { RSA_free(p);})
#define use_shared_openssl_pkey(pkey) std::shared_ptr<EVP_PKEY>(pkey, [](EVP_PKEY* p) { EVP_PKEY_free(p); })
#define use_shared_openssl_x509(cert) std::shared_ptr<X509>(cert, [](x509* cert) { X509_free(cert); })
#define use_shared_openssl_bn(bn) std::shared_ptr<BIGNUM>(bn, [](BIGNUM* p) { BN_free(p);})
#define use_shared_openssl_rsameth(meth) std::shared_ptr<RSA_METHOD>(meth, [](RSA_METHOD* p) { RSA_meth_free(p);})

class SecureConnect {
public:
    SecureConnect();
    ~SecureConnect();

    void setPKISdkType(int sdk_type);
    void SetRootcaCertBundle(std::string rootca_cert_bundle);
    void SetClientAuthParamWithAp(std::string client_cert, std::string client_priv_slot);
    void SetClientAuthParamWithOpenssl(std::string client_cert, std::string client_priv_key);
    void SetClientAuthP12ParamWithOpenssl(std::string client_key_cert_p12, std::string client_key_cert_p12_pass);
    void SetSslCtx(void *sslctx); // TODO RENAME

private:
    int VerifyCertsWithAp(X509_STORE_CTX *ctx);
    int VerifyCertsWithOpenssl(X509_STORE_CTX *ctx);
    bool ValidateEndCert(X509* cert);
    bool ValidateWithRoot(X509* cert,X509_STORE_CTX *ctx);
    // void LogCertInfo(X509* cert);
    std::string GetSubjectCommonName(X509* cert);
    std::string GetIssuerCommonName(X509* cert);
    time_t GetTimeNoBefore(X509* cert);
    time_t GetTimeNoAfter(X509* cert);
    int GetCertVersion(X509* cert);
    void ConfigureServerAuthWithAp(void *sslctx);
    void ConfigureClientAuthWithAp(void *sslctx, std::string priv_slot_uuid);
    void ConfigureServerAuthWithOpenssl(void *sslctx);
    void ConfigureClientAuthWithOpenssl(void *sslctx, std::string client_cert_chain_file, std::string priv_slot_key);
    void ConfigureClientAuthByP12WithOpenssl(void* sslctx, std::string client_key_p12, std::string pass);
    // openssl callback
    static int SslAppVerifyWithApCallback(X509_STORE_CTX *ctx, void *arg);
    static int SslAppVerifyWithOpensslCallback(X509_STORE_CTX *ctx, void *arg);
    // std::shared_ptr<EVP_PKEY> MakePrivateKeyProxy(std::string slot_uuid);
    std::string OpensslErrorString();

    // RSA encryption method for private key.
    // static int RsaPrivEnc(int flen, const uint8_t *from, unsigned char *to, RSA *rsa, int padding);
    // RSA decryption method for private key.
    // static int RsaPrivDec(int flen, const uint8_t *from, unsigned char *to, RSA *rsa, int padding);
    static int RsaPrivSign(int type, const unsigned char *m,
                                   unsigned int m_length,
                                   unsigned char *sigret, unsigned int *siglen,
                                   const RSA *rsa);
    static int RsaPrivVerify(int dtype, const unsigned char *m,
                                       unsigned int m_length,
                                       const unsigned char *sigbuf,
                                       unsigned int siglen, const RSA *rsa);

    // static int EvpRsaSign(EVP_PKEY_CTX *ctx, unsigned char *sig, size_t *siglen, const unsigned char *tbs, size_t tbslen);
    static void MessageCallback(int write_p, int version, int content_type, const void *buf, size_t len, SSL *ssl, void *arg);
    static void SslInfoCallback(const SSL *ssl, int type, int val);
    struct TlsProtocolMsgContainer {
        int protocol_msg_len_cur_ = 0;
        std::vector<uint8_t> protocol_msg_buffer_;
    };

    struct SslAppData {
        std::string slot_uuid_str_;
        std::shared_ptr<TlsProtocolMsgContainer> tls_protocol_msg_container_;
        // ~PrivateKeyData() {
        //     tls_protocol_msg_container_ = nullptr;
        // }
    };

    int sdkType;
    std::string rootca_cert_bundle_;
    std::string client_cert_;
    std::string client_key_cert_p12_;
    std::string client_key_cert_p12_pass_;
    std::string client_priv_key_;
    std::string client_priv_slot_;
    static bool pkey_rsa_method_set_;
};

}
}
}
#endif
