#include "x509/cimp_cert_sign_request.h"

#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include "common/crypto_logger.hpp"
#include "cryp/cimpl_signer_private_ctx.h"
#include "cryp/cryobj/cimpl_public_key.h"
#include "cryp/cryobj/cimpl_private_key.h"
#include "x509/cimp_x509_dn.h"
#include "x509/x509_provider.h"
#include "x509/x509_dn.h"
#include "x509/cimp_x509_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

using namespace hozon::netaos::crypto::cryp;
extern std::map<X509DN::AttributeId, std::string> AttributeId_NID_MAP;

CimpCertSignRequest::CimpCertSignRequest(const CmX509_Ref& ref)
: csr_ref_(ref) {

}

bool CimpCertSignRequest::Verify() { return false; }

std::vector<uint8_t> CimpCertSignRequest::ExportASN1CertSignRequest() {
    CRYP_INFO << "X509CmClient ExportASN1CertSignRequest";
    std::vector<uint8_t> csr_vec;
    X509CmClient::Instance().ExportCSR(this->csr_ref_, csr_vec);
    return csr_vec;
}

// const hozon::netaos::crypto::cryp::Signature& CertSignRequest::GetSignature() 
// {
//     return signature;
// }

unsigned CimpCertSignRequest::Version() 
{
    // return X509_REQ_VERSION_1; 
    return 0;
}

uint32_t CimpCertSignRequest::GetPathLimit() {
    return 0;
}

bool CimpCertSignRequest::IsCa() {
    return false;
}

const X509DN& CimpCertSignRequest::SubjectDn() {
    return x509dn_;
}

CimpCertSignRequest::CimpCertSignRequest() {
    // x509_req = X509_REQ_new();
    // signature = new Signature();
}

CimpCertSignRequest::~CimpCertSignRequest() {}


X509Provider& CimpCertSignRequest::MyProvider() {
    X509Provider* prov = new CimpX509Provider;
    return *prov;
}

}
}
}
}  // namespace hozon