#ifndef CIMP_CERT_SIGN_REQUEST_H_
#define CIMP_CERT_SIGN_REQUEST_H_

#include <openssl/x509.h>

#include "cryp/signer_private_ctx.h"
#include "x509/basic_cert_info.h"
#include "x509/cert_sign_request.h"
#include "client/x509_cm_client.h"
#include "x509/cimp_x509_dn.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class CimpCertSignRequest : public CertSignRequest {
public:
    CimpCertSignRequest(const CmX509_Ref& ref);
    bool Verify () override;
    std::vector<uint8_t> ExportASN1CertSignRequest() override;

    // virtual const hozon::netaos::crypto::cryp::Signature& GetSignature ();
    unsigned Version () override;
    uint32_t GetPathLimit() override;
    bool IsCa() override;
    const X509DN& SubjectDn() override;
    CimpCertSignRequest();
    ~CimpCertSignRequest();
    virtual X509Provider& MyProvider();

    CmX509_Ref getCsrRef() {
        return csr_ref_;
    }

private:
    mutable CmX509_Ref csr_ref_;
    x509::CimpX509DN x509dn_;

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon

#endif  // CERT_SIGN_REQUEST_H
