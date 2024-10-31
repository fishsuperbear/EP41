/*
 * Copyright (c) hozonauto. 2021-2021. All rights reserved.
 * Description: Http common definition
 */

#ifndef V2C_SECUREHTTP_SECURE_HTTP_ERR_H
#define V2C_SECUREHTTP_SECURE_HTTP_ERR_H

namespace hozon {
namespace netaos {
namespace https {
enum HttpsErr {
    kNoErr = 0,

    kErrHttpStart = 1,

    kErrTlsStart = 30,

    kErrCryptoStart = 50,
    kErrX509CertSignatureFailure = 50,
    kErrX509CertExpired,
    kErrX509CertFuture,
    kErrX509NoIssuerCert,
    kErrX509CertParseErr,
    kErrX509CertStatusUnknown,
    kErrX509CPFault,
};
}
}  // namespace v2c
}  // namespace hozon

#endif