#ifndef IMP_OCSP_RESPONSE_H
#define IMP_OCSP_RESPONSE_H

// #include "crypto/crypto_service/include/certificate_management/x509_object.h"
#include "x509/x509_object.h"
#include "x509/ocsp_response.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class ImpOcspResponse : public OcspResponse {
public:
    using Uptrc = std::unique_ptr<const OcspResponse>;
    using Uptr = std::unique_ptr<OcspResponse>;

    ImpOcspResponse(int status) {
        respStatus = status;
    }

    virtual std::uint32_t Version () const;

    virtual std::uint32_t RespStatus () const;
    virtual ~ImpOcspResponse(){};

    virtual X509Provider& MyProvider();

   private:
};

}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // OCSP_RESPONSE_H
