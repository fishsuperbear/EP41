#ifndef CERT_SIGN_REQUEST_H
#define CERT_SIGN_REQUEST_H

// TODO: DONOT USE OPENSSL IN HEADER FILE.
// #include <openssl/x509.h>
#include "x509/basic_cert_info.h"
#include "cryp/signer_private_ctx.h"
#include "cryp/cryobj/private_key.h"
#include "cryp/cryobj/public_key.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class CertSignRequest : public BasicCertInfo {
public:
    using Uptr = std::unique_ptr<CertSignRequest>;
    using Uptrc = std::unique_ptr<const CertSignRequest>;
    // X509_REQ *x509_req = nullptr;
    std::map<std::uint32_t, std::string> x509Extensions;
    std::uint32_t version;
    cryp::SignerPrivateCtx::Uptr signerCtx;
    cryp::PublicKey* pubkey;
    cryp::PrivateKey* pvkey;
    // todo: 依赖crypt
    /// @brief Verifies self-signed signature of the certificate request
    /// @return 
    virtual bool Verify () = 0;

    /// @brief Export this certificate signing request in DER encoded ASN1 format
    /// @return 
    virtual std::vector<uint8_t> ExportASN1CertSignRequest() = 0;

    // todo: 依赖crypt
    /// @brief Return signature object of the request
    /// @return 
    // virtual const hozon::netaos::crypto::cryp::Signature& GetSignature ();

    /// @brief Return format version of the certificate request.
    /// @return 
    virtual unsigned Version () = 0;

    // todo: 构造函数里初始化成员变量
    // CertSignRequest();
    virtual ~CertSignRequest() {};

private:
    // int add_ext(STACK_OF(X509_EXTENSION)* sk, X509_REQ * req, int nid, const char * value);

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // CERT_SIGN_REQUEST_H
