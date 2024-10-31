#ifndef X509_PROVIDER_H
#define X509_PROVIDER_H

#include <mutex>
#include <memory>
#include <map>

#include "core/vector.h"
#include "common/serializable.h"
#include "x509/ocsp_request.h"
#include "x509/ocsp_response.h"
#include "x509/cert_sign_request.h"
#include "x509/certificate.h"
#include "cryp/signer_private_ctx.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

using namespace hozon::netaos::core;

#define TEST_MODEL true
#define ROOT_CERT_STORAGE_PATH       "/cfg/pki/certs/root_ca/ecu_ca.pem"
#define CERT_CRL_STORAGE_PATH        "/cfg/pki/certs/server.crl"


class X509Provider {
public:
    using Uptr = std::unique_ptr<X509Provider>;
    using StorageIndex = std::size_t;
    void Init();
    void DeInit();
    virtual ~X509Provider() {};

    /// @brief Parse a serialized representation of the certificate and create its instance.
    /// @param cert DER/PEM-encoded certificate
    /// @param formatId input format identifier
    /// @return
    virtual Certificate::Uptr ParseCert(ReadOnlyMemRegion certMem, Serializable::FormatId formatId) = 0;

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
        std::uint32_t version=0) = 0;

    /// @brief Import the certificate to volatile or persistent storage.
    /// @param certPath a valid certificate that should be imported
    /// @param iSpecify optionally a valid InstanceSpecifier can be provided
                    // that points to a CertificateSlot for persistent storage
                    // of the certificate, otherwise the certificate shall be
                    // stored in volatile (session) storage
    /// @return
    virtual bool ImportCert(const Certificate::Uptr &cert, const std::string destCertPath = "") = 0;

    /// @brief Import Certificate Revocation List (CRL) or Delta CRL from a memory BLOB.
    /// @param crlPath serialized CRL or Delta CRL (in form of a BLOB)
    /// @return true if the CRL is valid and false if it is already xpired
    virtual bool ImportCrl(const std::string crlPath) = 0;

    virtual bool SetAsRootOfTrust(const Certificate::Uptr &caCert) = 0;

    virtual Certificate::Status VerifyCert(Certificate::Uptr &cert,
        const std::string rootCertPath = ROOT_CERT_STORAGE_PATH) = 0;

    virtual OcspRequest::Uptr CreateOcspRequest(const std::string certPath, const std::string rootPath = "") = 0;

    virtual Certificate::Status CheckCertStatus(const std::string certPath, const std::string ocspResponsePath, const std::string rootPath) = 0;

    virtual OcspResponse::Uptr ParseOcspResponse (const std::string ocspResponsePath) = 0;

    virtual Certificate::Uptr FindCertByDn(const X509DN &subjectDn,
        const X509DN &issuerDn, time_t validityTimePoint) noexcept = 0;

    virtual X509DN::Uptr BuildDn(std::string dn) noexcept = 0;
};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_PROVIDER_H
