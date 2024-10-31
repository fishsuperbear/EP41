#ifndef IMP_CERTIFICATE_H
#define IMP_CERTIFICATE_H

#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include "x509/basic_cert_info.h"
#include "x509/certificate.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class ImpCertificate : public Certificate {
public:
    ImpCertificate();

    virtual ~ImpCertificate();

    virtual BasicCertInfo::KeyConstraints GetConstraints();

    virtual uint32_t GetPathLimit();

    virtual bool IsCa();

    virtual const X509DN& SubjectDn();

    //  @brief Transmit a Uds message via the underlying Uds Transport Protocol channel.
    //
    //  @param message Const ptr of reply udsmessage.
    //  @param transmitChannelId Channel ID.
    //  @return void
    virtual std::string AuthorityKeyId();

    /// @brief Get the "Not Before" of the certificate
    /// @return time_t
    virtual time_t StartTime();

    /// @brief Get the "Not After" of the certificate.
    /// @return time_t
    virtual time_t EndTime();

    // virtual hozon::netaos::core::Result<std::size_t> GetFingerprint (ReadWriteMemRegion fingerprint, cryp::HashFunctionCtx &hashCtx) const noexcept=0;

    /// @brief Return last verification status of the certificate
    /// @return Status
    virtual Certificate::Status GetStatus();

    /// @brief Check whether this certificate belongs to a root CA.
    /// @return bool
    virtual bool IsRoot();

    /// @brief Get the issuer certificate distinguished name).
    /// @return X509DNImpCertificate
    virtual const X509DN& IssuerDn();

    /// @brief Get the serial number of this certificate
    /// @return buffer
    virtual std::string SerialNumber();

    /// @brief Get the DER encoded SubjectKeyIdentifier of this certificate
    /// @return buffer
    virtual std::string SubjectKeyId();

    /// @brief Verify signature of the certificate
    /// @param caCert
    /// @return bool
    virtual bool VerifyMe(Certificate::Uptr caCert) ;

    /// @brief Get the X.509 version of this certificate object
    /// @return uint32_t
    virtual std::uint32_t X509Version();

    virtual X509Provider& MyProvider();

    // TODO
    X509* cert_;
private:
};

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // CERTIFICATE_H
