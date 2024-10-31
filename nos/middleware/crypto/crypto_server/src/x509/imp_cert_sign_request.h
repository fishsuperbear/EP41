#ifndef IMP_CERT_SIGN_REQUEST_H
#define IMP_CERT_SIGN_REQUEST_H

#include <openssl/x509.h>

#include "cryp/signer_private_ctx.h"
#include "x509/basic_cert_info.h"
#include "x509/cert_sign_request.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class ImpCertSignRequest : public CertSignRequest {
public:

    virtual uint32_t GetPathLimit();

    virtual bool IsCa();

    virtual const X509DN& SubjectDn();

    // todo: 依赖crypt
    /// @brief Verifies self-signed signature of the certificate request
    /// @return 
    virtual bool Verify ();

    /// @brief Export this certificate signing request in DER encoded ASN1 format
    /// @return 
    virtual std::vector<uint8_t> ExportASN1CertSignRequest();

    // todo: 依赖crypt
    /// @brief Return signature object of the request
    /// @return 
    // virtual const hozon::netaos::crypto::cryp::Signature& GetSignature ();

    /// @brief Return format version of the certificate request.
    /// @return 
    virtual unsigned Version ();

    // todo: 构造函数里初始化成员变量
    ImpCertSignRequest();
    virtual ~ImpCertSignRequest();

    virtual X509Provider& MyProvider();

private:
    int add_ext(STACK_OF(X509_EXTENSION)* sk, X509_REQ * req, int nid, const char * value);

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // CERT_SIGN_REQUEST_H
