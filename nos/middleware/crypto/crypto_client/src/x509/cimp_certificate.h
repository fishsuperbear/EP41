#ifndef CIMP_CERTIFICATE_H
#define CIMP_CERTIFICATE_H

#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include "x509/basic_cert_info.h"
#include "x509/certificate.h"
#include "common/inner_types.h"
#include "x509/cimp_x509_dn.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class CimpCertificate : public Certificate {
public:
    CimpCertificate();
    CimpCertificate(CryptoCertRef cert_ref):cert_ref_(cert_ref),subject_dn_(nullptr),issuer_dn_(nullptr){}

    ~CimpCertificate(){
        if(subject_dn_){
            delete subject_dn_;
            subject_dn_ = nullptr;
        }

        if(issuer_dn_){
            delete issuer_dn_;
            issuer_dn_ = nullptr;
        }
    };

    // BasicCertInfo::KeyConstraints GetConstraints();

    uint32_t GetPathLimit();

    bool IsCa();

    const X509DN& SubjectDn();

    std::string AuthorityKeyId();

    /// @brief Get the "Not Before" of the certificate
    /// @return time_t
    time_t StartTime();

    /// @brief Get the "Not After" of the certificate.
    /// @return time_t
    time_t EndTime();

    // virtual hozon::netaos::core::Result<std::size_t> GetFingerprint (ReadWriteMemRegion fingerprint, cryp::HashFunctionCtx &hashCtx) const noexcept=0;

    /// @brief Return last verification status of the certificate
    /// @return Status
    Certificate::Status GetStatus();

    /// @brief Check whether this certificate belongs to a root CA.
    /// @return bool
    bool IsRoot();

    /// @brief Get the issuer certificate distinguished name).
    /// @return X509DNImpCertificate
    const X509DN& IssuerDn();

    /// @brief Get the serial number of this certificate
    /// @return buffer
    std::string SerialNumber();

    /// @brief Get the DER encoded SubjectKeyIdentifier of this certificate
    /// @return buffer
    std::string SubjectKeyId();

    /// @brief Verify signature of the certificate
    /// @param caCert
    /// @return bool
    bool VerifyMe(Certificate::Uptr caCert) ;

    /// @brief Get the X.509 version of this certificate object
    /// @return uint32_t
    std::uint32_t X509Version();

    X509Provider& MyProvider();

    CryptoCertRef GetCertRef(){
        return cert_ref_;
    }
    // TODO
    // X509* cert_;
private:
    CryptoCertRef cert_ref_;
    CimpX509DN *subject_dn_;
    CimpX509DN *issuer_dn_;
    std::unique_ptr<X509Provider> my_provider_ = nullptr;

};

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // CERTIFICATE_H
