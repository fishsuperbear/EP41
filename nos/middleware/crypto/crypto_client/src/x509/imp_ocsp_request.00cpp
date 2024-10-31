#include "x509/imp_ocsp_request.h"

#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/ocsp.h>

#include "common/crypto_logger.hpp"
#include "x509/ocsp_request.h"
#include "x509/x509_provider.h"
#include "x509/imp_x509_provider.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

unsigned ImpOcspRequest::Version() const {
    return version;
}

bool ImpOcspRequest::ExportASN1OCSPRequest(std::string saveOcspFilePath) {
    CRYP_ERROR << "ExportASN1OCSPRequest cert: "<< this->certPath;
    CRYP_ERROR << "ExportASN1OCSPRequest cert: "<< this->rootPath;
    OCSP_REQUEST *req = nullptr;
    OCSP_CERTID* id = nullptr;
    OCSP_ONEREQ* oneReq = nullptr;
    BIO* bio = nullptr;
    X509* subject = nullptr;
    X509* issuer = nullptr;

    auto PTR_FREE = [&]() -> void {
        if (req != nullptr) {
            // 释放 OCSO REQUEST
            OCSP_REQUEST_free(req);
        }
        if (id != nullptr) {
            // 释放 OCSP CERTID
            OCSP_CERTID_free(id);
        }
        if (subject != nullptr) {
            // 释放 X509
            X509_free(subject);
        }
        if (issuer != nullptr) {
            // 释放 X509
            X509_free(issuer);
        }
        if (oneReq != nullptr) {
            // 释放 X509 CRL
            OCSP_ONEREQ_free(oneReq);
        }
        if (bio != nullptr) {
            // 释放 BIO
            BIO_free(bio);
        }
    };
    // load signed cert
    subject = dynamic_cast<ImpX509Provider*>(&MyProvider())->LoadX509Cert(this->certPath);
    if (subject == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "LoadX509Cert subject failed";
        return false;
    }

    // load root cert
    if (this->rootPath.empty()) {
        issuer = dynamic_cast<ImpX509Provider*>(&MyProvider())->LoadX509Cert(ROOT_CERT_STORAGE_PATH);
    } else {
        issuer = dynamic_cast<ImpX509Provider*>(&MyProvider())->LoadX509Cert(this->rootPath);
    }
    if (subject == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "LoadX509Cert issuer failed";
        return false;
    }
    // 根据摘要算法、持有者证书和颁发者证书生成OCSP_CERTID
    id = OCSP_cert_to_id(nullptr, subject, issuer);

    req = OCSP_REQUEST_new();
    if (req == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "OCSP_REQUEST_new failed";
        return false;
    }
    // 添加证书请求
    if ( !OCSP_request_add0_id(req, id) ) {
            CRYP_ERROR << "creat OCSP_REQUEST failed";
            return false;
    }

    bio = BIO_new_file(saveOcspFilePath.c_str(), "wb");
    if (bio == nullptr) {
        CRYP_ERROR << "creat OCSP_REQUEST BIO_new_file bio is null";
        return false;
    }
    // DER编码并写入文件
    if (ASN1_i2d_bio(reinterpret_cast<i2d_of_void *>(i2d_OCSP_REQUEST), bio, req) <= 0) {
        CRYP_ERROR << "creat OCSP_REQUEST ASN1_i2d_bio failed";
        return false;
    }
    return true;
}

X509Provider& ImpOcspRequest::MyProvider() {
    X509Provider* prov = new ImpX509Provider;
    return *prov;
}

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
