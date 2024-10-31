#ifndef IMP_X509_PROVIDER_H
#define IMP_X509_PROVIDER_H

#include <mutex>
#include <memory>
#include <map>

#include "common/serializable.h"
#include "cryp/imp_signer_private_ctx.h"
#include "x509/imp_cert_sign_request.h"
#include "x509/imp_ocsp_request.h"
#include "x509/imp_ocsp_response.h"
#include "x509/imp_certificate.h"
#include "x509/x509_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

using namespace hozon::netaos::core;

class ImpX509Provider : public X509Provider{
public:
    // static X509Provider* getInstance();
    void Init();
    void DeInit();
    virtual ~ImpX509Provider();

    /// @brief Parse a serialized representation of the certificate and create its instance.
    /// @param cert DER/PEM-encoded certificate
    /// @param formatId input format identifier
    /// @return
    Certificate::Uptr ParseCert(ReadOnlyMemRegion certMem, Serializable::FormatId formatId);


    /// @brief Create certification request for a private key loaded to the context.
    /// @param signerCtx 
    /// @param derSubjectDN 
    /// @param x509Extensions 
    /// @param version 
    /// @return 
    // virtual CertSignRequest::Uptr CreateCertSignRequest(
    //     cryp::SignerPrivateCtx::Uptr signerCtx, 
    //     std::string &derSubjectDN, 
    //     std::string &x509Extensions,
    //     std::uint32_t version=2);
    virtual CertSignRequest::Uptr CreateCertSignRequest(
        cryp::SignerPrivateCtx::Uptr& signerCtx,
        X509DN::Uptr& derSubjectDN,
        std::map<std::uint32_t, std::string> &x509Extensions,
        std::uint32_t version=0);

    /// @brief Import the certificate to volatile or persistent storage.
    /// @param certPath a valid certificate that should be imported
    /// @param iSpecify optionally a valid InstanceSpecifier can be provided
                    // that points to a CertificateSlot for persistent storage
                    // of the certificate, otherwise the certificate shall be
                    // stored in volatile (session) storage
    /// @return
    virtual bool ImportCert(const Certificate::Uptr &cert, const std::string destCertPath = "");

    /// @brief Import Certificate Revocation List (CRL) or Delta CRL from a memory BLOB.
    /// @param crlPath serialized CRL or Delta CRL (in form of a BLOB)
    /// @return true if the CRL is valid and false if it is already xpired
    virtual bool ImportCrl(const std::string crlPath);

    virtual bool SetAsRootOfTrust(const Certificate::Uptr &caCert);

    virtual Certificate::Status VerifyCert(Certificate::Uptr &cert,
        const std::string rootCertPath = ROOT_CERT_STORAGE_PATH);

    virtual OcspRequest::Uptr CreateOcspRequest(const std::string certPath, const std::string rootPath = "");

    virtual Certificate::Status CheckCertStatus(const std::string certPath, const std::string ocspResponsePath, const std::string rootPath);

    virtual OcspResponse::Uptr ParseOcspResponse (const std::string ocspResponsePath);

    virtual X509DN::Uptr BuildDn(std::string dn) noexcept;

    virtual Certificate::Uptr FindCertByDn(const X509DN &subjectDn,
        const X509DN &issuerDn, time_t validityTimePoint) noexcept;

    X509* LoadX509Cert(const std::string certPath);
    X509_CRL* LoadX509Crl(const std::string crlPath);
    OCSP_RESPONSE* LoadOcspResponse(const std::string ocspResponsePath);

    BasicCertInfo::KeyConstraints ParseOpensslConstraints(uint32_t sslKeyUsage);
};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_PROVIDER_H
