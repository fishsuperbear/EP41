#ifndef CERTIFICATE_H
#define CERTIFICATE_H
// #include <openssl/x509.h>
// #include <openssl/x509v3.h>
#include "basic_cert_info.h"
namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class Certificate : public BasicCertInfo {
   public:
    using Uptr = std::unique_ptr<Certificate>;
    using Uptrc = std::unique_ptr<const Certificate>;

    X509DN::Uptr subjectDn_uptr;
    X509DN::Uptr issuerDn_uptr;
    enum class Status : std::uint32_t { kValid = 0, kInvalid = 1, kUnknown = 2, kNoTrust = 3, kExpired = 4, kFuture = 5 };
    // kValid= 0 The certificate is valid.
    // kInvalid= 1 The certificate is invalid.
    // kUnknown= 2 Status of the certificate is unknown yet.
    // kNoTrust= 3 The certificate has correct signature, but the ECU has no a root of trust for this certificate.
    // kExpired= 4 The certificate has correct signature, but it is already expired (its validity period has ended)
    // kFuture= 5 The certificate has correct signature, but its validity period is not started yet.

    Serializable::FormatId formatId;

    // Certificate();
    virtual ~Certificate() {};

    // virtual BasicCertInfo::KeyConstraints GetConstraints();

    // virtual uint32_t GetPathLimit() = 0;

    // virtual bool IsCa() = 0;

    // virtual const X509DN& SubjectDn() = 0;

    //  @brief Transmit a Uds message via the underlying Uds Transport Protocol channel.
    //
    //  @param message Const ptr of reply udsmessage.
    //  @param transmitChannelId Channel ID.
    //  @return void
    virtual std::string AuthorityKeyId() = 0;

    /// @brief Get the "Not Before" of the certificate
    /// @return time_t
    virtual time_t StartTime() = 0;

    /// @brief Get the "Not After" of the certificate.
    /// @return time_t
    virtual time_t EndTime() = 0;

    // virtual hozon::netaos::core::Result<std::size_t> GetFingerprint (ReadWriteMemRegion fingerprint, cryp::HashFunctionCtx &hashCtx) const noexcept=0;

    /// @brief Return last verification status of the certificate
    /// @return Status
    virtual Status GetStatus() = 0;

    /// @brief Check whether this certificate belongs to a root CA.
    /// @return bool
    virtual bool IsRoot() = 0;

    /// @brief Get the issuer certificate distinguished name).
    /// @return X509DN
    virtual const X509DN& IssuerDn() = 0;

    /// @brief Get the serial number of this certificate
    /// @return buffer
    virtual std::string SerialNumber() = 0;

    /// @brief Get the DER encoded SubjectKeyIdentifier of this certificate
    /// @return buffer
    virtual std::string SubjectKeyId() = 0;

    /// @brief Verify signature of the certificate
    /// @param caCert
    /// @return bool
    virtual bool VerifyMe(Certificate::Uptr caCert) = 0;

    /// @brief Get the X.509 version of this certificate object
    /// @return uint32_t
    virtual std::uint32_t X509Version() = 0;
};

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // CERTIFICATE_H
