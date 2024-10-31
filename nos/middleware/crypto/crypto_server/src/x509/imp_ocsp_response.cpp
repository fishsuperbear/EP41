#include "x509/imp_ocsp_response.h"

#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/ocsp.h>
#include <openssl/pem.h>

#include "common/crypto_logger.hpp"
#include "x509/ocsp_response.h"
#include "x509/imp_x509_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

unsigned ImpOcspResponse::Version() const {
    return version;
}

unsigned ImpOcspResponse::RespStatus() const {
    return respStatus;
}

X509Provider& ImpOcspResponse::MyProvider() {
    X509Provider* prov = new ImpX509Provider;
    return *prov;
}

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon

