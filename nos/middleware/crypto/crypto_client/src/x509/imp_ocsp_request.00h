#ifndef IMP_OCSP_REQUEST_H
#define IMP_OCSP_REQUEST_H

#include "x509/x509_object.h"
#include "x509/ocsp_request.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class ImpOcspRequest : public OcspRequest {
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

    ImpOcspRequest(const std::string cert, const std::string root):OcspRequest(cert, root){

    }
    virtual ~ImpOcspRequest(){};
    virtual std::uint32_t Version () const;

    bool ExportASN1OCSPRequest(std::string saveOcspFilePath);

    virtual X509Provider& MyProvider();

private:
};


}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // OCSP_REQUEST_H
