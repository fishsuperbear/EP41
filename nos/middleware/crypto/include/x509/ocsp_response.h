#ifndef OCSP_RESPONSE_H
#define OCSP_RESPONSE_H

// #include "crypto/crypto_service/include/certificate_management/x509_object.h"
#include "x509_object.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {
class OcspResponse : public X509Object {
public:
    using Uptrc = std::unique_ptr<const OcspResponse>;
    using Uptr = std::unique_ptr<OcspResponse>;

    // OcspResponse(int status):respStatus(status) {}

    virtual std::uint32_t Version () const  = 0;

    virtual std::uint32_t RespStatus () const  = 0;
    virtual ~OcspResponse(){};

   protected:
    std::uint32_t version;
    int respStatus;

};

}  // namesaoce x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
#endif  // OCSP_RESPONSE_H
