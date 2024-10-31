#include "common/crypto_logger.hpp"
#include "x509/certificate.h"
#include "x509/cimp_x509_dn.h"
#include "x509/cimp_x509_provider.h"
#include "cimp_certificate.h"
#include "client/x509_cm_client.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

// BasicCertInfo::KeyConstraints CimpCertificate::GetConstraints() {

//     return X509CmClient::Instance().GetConstraints(cert_ref_.ref);

// }

uint32_t CimpCertificate::GetPathLimit() {
    //  return X509CmClient::Instance().GetPathLimit(cert_ref_);
    return 100;
}

bool CimpCertificate::IsCa() {
    return X509CmClient::Instance().IsCa(cert_ref_);
}

const X509DN& CimpCertificate::SubjectDn() {
    auto ret =  X509CmClient::Instance().SubjectDn(cert_ref_);
    subject_dn_ = new CimpX509DN(ret);
    return *subject_dn_;
}

std::string CimpCertificate::AuthorityKeyId(){
    return X509CmClient::Instance().AuthorityKeyId(cert_ref_);
}

const X509DN& CimpCertificate::IssuerDn() {
    auto ret =  X509CmClient::Instance().IssuerDn(cert_ref_);
    issuer_dn_ = new CimpX509DN(ret);
    return *issuer_dn_;
}

std::uint32_t CimpCertificate::X509Version()
{
    return  X509CmClient::Instance().X509Version(cert_ref_);
}


std::string CimpCertificate::SubjectKeyId() {
    return  X509CmClient::Instance().SubjectKeyId(cert_ref_);
}

time_t CimpCertificate::StartTime() {
    return  X509CmClient::Instance().StartTime(cert_ref_);
}

time_t CimpCertificate::EndTime() {
    return  X509CmClient::Instance().EndTime(cert_ref_);
}

Certificate::Status CimpCertificate::GetStatus() {
    return  X509CmClient::Instance().GetStatus(cert_ref_);
}

bool CimpCertificate::IsRoot() {
    return  X509CmClient::Instance().IsRoot(cert_ref_);
}

std::string CimpCertificate::SerialNumber() {
    return  X509CmClient::Instance().SerialNumber(cert_ref_);
}

bool CimpCertificate::VerifyMe(Certificate::Uptr caCert) {
    // todo: 依赖密码学原语
    return false; 
}

X509Provider& CimpCertificate::MyProvider() {
    if (!my_provider_) {
        my_provider_ = std::make_unique<CimpX509Provider>();
    }
    return *my_provider_.get();
}

CimpCertificate::CimpCertificate() {}


}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
