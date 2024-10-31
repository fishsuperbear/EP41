#ifndef CIMP_X509_PROVIDER_H
#define CIMP_X509_PROVIDER_H

#include <mutex>
#include <memory>
#include <map>

#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/ocsp.h>

#include "common/serializable.h"
// #include "cryp/cimpl_signer_private_ctx.h"
// #include "x509/cimp_cert_sign_request.h"
// #include "x509/imp_ocsp_request.h"
// #include "x509/imp_ocsp_response.h"
// #include "x509/cimp_certificate.h"
#include "x509/x509_provider.h"
#include "common/inner_types.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

using namespace hozon::netaos::core;

#define TEST_MODEL true

class CimpX509Provider : public X509Provider{
public:
    // static X509Provider* getInstance();
    void Init();
    void DeInit();
    CimpX509Provider();
    ~CimpX509Provider();
    Certificate::Uptr ParseCert(ReadOnlyMemRegion certMem, Serializable::FormatId formatId);
    CertSignRequest::Uptr CreateCertSignRequest(
        cryp::SignerPrivateCtx::Uptr& signerCtx,
        X509DN::Uptr& derSubjectDN,
        std::map<std::uint32_t, std::string> &x509Extensions,
        std::uint32_t version=0);

    bool ImportCert(const Certificate::Uptr &cert, const std::string destCertPath = "");
    bool ImportCrl(const std::string crlPath);
    bool SetAsRootOfTrust(const Certificate::Uptr &caCert);

    Certificate::Status VerifyCert(Certificate::Uptr &cert,const std::string rootCertPath = ROOT_CERT_STORAGE_PATH);
    OcspRequest::Uptr CreateOcspRequest(const std::string certPath, const std::string rootPath = "");
    Certificate::Status CheckCertStatus(const std::string certPath, const std::string ocspResponsePath, const std::string rootPath);
    OcspResponse::Uptr ParseOcspResponse (const std::string ocspResponsePath);

    X509DN::Uptr BuildDn(std::string dn) noexcept;

    Certificate::Uptr FindCertByDn(const X509DN &subjectDn,
        const X509DN &issuerDn, time_t validityTimePoint) noexcept;

    X509* LoadX509Cert(const std::string certPath);
    X509_CRL* LoadX509Crl(const std::string crlPath);
    OCSP_RESPONSE* LoadOcspResponse(const std::string ocspResponsePath);
    BasicCertInfo::KeyConstraints ParseOpensslConstraints(uint32_t sslKeyUsage);

private:
    X509DNRef dn_ref_;
    CryptoCertRef cert_ref_;

};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // X509_PROVIDER_H
