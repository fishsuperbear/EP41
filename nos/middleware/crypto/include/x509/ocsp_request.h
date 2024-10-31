#ifndef OCSP_REQUEST_H
#define OCSP_REQUEST_H

#include "x509_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class OcspRequest : public X509Object {
public:
    using Uptrc = std::unique_ptr<const OcspRequest>;
    using Uptr = std::unique_ptr<OcspRequest>;
    enum class OcspResponseStatus : uint32_t {
        kSuccessful= 0,              // Response has valid confirmations.
        kMalformedRequest= 1,        // Illegal confirmation request.
        kInternalError= 2,           // Internal error in issuer.
        kTryLater= 3,                // Try again later.
        kSigRequired= 5,             // Must sign the request.
        kUnauthorized= 6
    };

    OcspRequest(std::string cert, std::string root)
    :certPath(cert),
    rootPath(root) {
    }
    virtual ~OcspRequest(){};
    virtual std::uint32_t Version () const  = 0;

    virtual bool ExportASN1OCSPRequest(std::string saveOcspFilePath) = 0;

protected:
    std::uint32_t version;
    const std::string certPath;
    const std::string rootPath;
};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // OCSP_REQUEST_H
