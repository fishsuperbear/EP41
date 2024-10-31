#include "x509/imp_certificate.h"

#include "common/crypto_logger.hpp"
#include "x509/certificate.h"
#include "x509/imp_x509_dn.h"
#include "x509/imp_x509_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

BasicCertInfo::KeyConstraints ImpCertificate::GetConstraints() {
    auto sslKeyUsage = X509_get_key_usage(this->cert_);
    if (UINT16_MAX == sslKeyUsage) {
        return 0;
    }

    if (0x8000 == sslKeyUsage) {
        return 0x80;
    }

    uint32_t ret = (sslKeyUsage << 8);
    return ret;
}

uint32_t ImpCertificate::GetPathLimit() {
    // 获取证书链长度限制
    return X509_get_pathlen(this->cert_);
}

bool ImpCertificate::IsCa() {
    BASIC_CONSTRAINTS* basic_constraints = nullptr;
    basic_constraints = reinterpret_cast<BASIC_CONSTRAINTS*>(X509_get_ext_d2i(this->cert_, NID_basic_constraints, NULL, NULL));
    if (basic_constraints != NULL) {
        int ca_flag = basic_constraints->ca;
        BASIC_CONSTRAINTS_free(basic_constraints);
        if (ca_flag) {
            CRYP_INFO << "IsCa, this cert is ca";
            return true;
        }
    }
    CRYP_INFO << "IsCa, this cert is not ca";
    return false;
}

const X509DN& ImpCertificate::SubjectDn() {
    X509_NAME* subject_dn = X509_get_subject_name(this->cert_);
    char subject_buf[4096];
    X509_NAME_oneline(subject_dn, subject_buf, sizeof(subject_buf));
    std::string str_subject(subject_buf);
    CRYP_INFO << "SubjectDn :" <<str_subject.c_str();
    subjectDn_uptr = this->MyProvider().BuildDn(str_subject);
    return *(subjectDn_uptr.get());
}

const X509DN& ImpCertificate::IssuerDn() {
    X509_NAME* issuer_dn = X509_get_issuer_name(this->cert_);
    char issuer_buf[4096];
    X509_NAME_oneline(issuer_dn, issuer_buf, sizeof(issuer_buf));
    std::string str_issuer(issuer_buf);
    CRYP_INFO << "IssuerDn :" <<str_issuer.c_str();
    issuerDn_uptr = this->MyProvider().BuildDn(str_issuer);
    return *(issuerDn_uptr.get());
}

std::uint32_t ImpCertificate::X509Version()
{
    return X509_get_version(this->cert_);
}

std::string ImpCertificate::AuthorityKeyId()
{
    std::string authorityKeyId;
    authorityKeyId.append(
        reinterpret_cast<const char*>(X509_get0_authority_key_id(this->cert_)->data));
    return authorityKeyId;
}

std::string ImpCertificate::SubjectKeyId() {
    std::string subjectKeyid;
    subjectKeyid.append(
        reinterpret_cast<const char*>(X509_get0_subject_key_id(this->cert_)->data));
    return subjectKeyid;
}

time_t ImpCertificate::StartTime() {
    struct tm notBefore;
    ASN1_TIME_to_tm(X509_getm_notBefore(this->cert_), &notBefore);
    return mktime(&notBefore);
}

time_t ImpCertificate::EndTime() {
    // 获取证书有效期
    struct tm notAfter;
    ASN1_TIME_to_tm(X509_getm_notAfter(this->cert_), &notAfter);
    return mktime(&notAfter);
}

Certificate::Status ImpCertificate::GetStatus() {
    // ???
    return Certificate::Status();
}

bool ImpCertificate::IsRoot() {
    if (IsCa()) {
        X509_NAME* subject_dn = X509_get_subject_name(this->cert_);
        char subject_buf[4096];
        X509_NAME_oneline(subject_dn, subject_buf, sizeof(subject_buf));
        std::string str_subject(subject_buf);
        // CRYP_INFO << "SubjectDn :" <<str_subject.c_str();

        X509_NAME* issuer_dn = X509_get_issuer_name(this->cert_);
        char issuer_buf[4096];
        X509_NAME_oneline(issuer_dn, issuer_buf, sizeof(issuer_buf));
        // CRYP_INFO << "issuerDn :" <<str_issuer.c_str();
        std::string str_issuer(issuer_buf);
        if (0 == str_subject.compare(str_issuer)) {
            CRYP_INFO << "IsRoot, this cert is root";
            return true;
        }
    }
    CRYP_INFO << "IsRoot, this cert is not root";
    return false;
}

std::string ImpCertificate::SerialNumber() {
    BIGNUM *serial_bn = nullptr;
    serial_bn = reinterpret_cast<BIGNUM*>(X509_get_serialNumber(this->cert_));
    char *serial_hex = BN_bn2hex(serial_bn);
    std::string serialNumber(serial_hex);
    return serialNumber;
}

bool ImpCertificate::VerifyMe(Certificate::Uptr caCert) {
    // todo: 依赖密码学原语
    return false; 
}

X509Provider& ImpCertificate::MyProvider() {
    X509Provider* prov = new ImpX509Provider;
    return *prov;
}

ImpCertificate::ImpCertificate() {}

ImpCertificate::~ImpCertificate() {
    // todo: 释放string里的内存！
}

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
