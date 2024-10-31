#include "x509/cimp_x509_provider.h"

#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/ocsp.h>
#include <dirent.h>

#include "common/crypto_logger.hpp"
#include "x509/x509_provider.h"
#include "x509/cimp_certificate.h"
#include "x509/cimp_cert_sign_request.h"
#include "x509/cimp_x509_dn.h"
#include "x509/cimp_x509_provider.h"
#include "cryp/cimpl_signer_private_ctx.h"
#include "client/x509_cm_client.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

X509* cert = nullptr;  // for LoadX509Cert from file path
X509_CRL* crl = nullptr;  // for LoadX509Crl from file path
OCSP_RESPONSE* ocspresp = nullptr;  // for Load ocsp response from file path

void CimpX509Provider::Init() {
    // CryptoLogger::GetInstance().InitLogging("crypto","crypto service",
    //     CryptoLogger::CryptoLogLevelType::CRYPTO_TRACE, //the log level of application
    //      hozon::netaos::log::HZ_LOG2FILE, //the output log mode
    //     "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
    //     10, //the max number log file , active when output log to file
    //     20 //the max size of each  log file , active when output log to file
    // );
    // CryptoLogger::GetInstance().CreateLogger("CRPS");
    // CRYP_DEBUG << "X509Provider::Init";
}

void CimpX509Provider::DeInit() {
    CRYP_DEBUG << "X509Provider::DeInit";
//   if (nullptr != instance_) {
//     delete instance_;
//     instance_ = nullptr;
//     CRYP_INFO << "delete X509Provider instance_";
//   }
}

CimpX509Provider::CimpX509Provider() {

}

CimpX509Provider::~CimpX509Provider() {

}

X509DN::Uptr CimpX509Provider::BuildDn(std::string dn) noexcept {
    CRYP_INFO << "X509CmClient BuildDn";
    int32_t ret = -1;
    ret = X509CmClient::Instance().BuildDn(dn,dn_ref_);
    if(ret == 0){
        X509DN::Uptr x509_dn_uptr = std::make_unique<CimpX509DN>(dn_ref_);
        return x509_dn_uptr;
    }else{
        CRYP_ERROR << " error";
        return X509DN::Uptr();
    }
}

Certificate::Uptr CimpX509Provider::ParseCert(ReadOnlyMemRegion certMem, Serializable::FormatId formatId) {
    // std::vector<const uint8_t> certdata(const_cast<uint8_t*>(certMem.data()),static_cast<std::size_t>(certMem.size()));
    std::vector<uint8_t> certdata;
    certdata.resize(certMem.size());
    memcpy(certdata.data(), certMem.data(), certMem.size());
    if(0 == X509CmClient::Instance().ParseCert(certdata,formatId,cert_ref_)){
        Certificate::Uptr cert_uptr(new CimpCertificate(cert_ref_));
        return cert_uptr;
    }else{
        return Certificate::Uptr();
    }
}


CertSignRequest::Uptr CimpX509Provider::CreateCertSignRequest(
    cryp::SignerPrivateCtx::Uptr& signerCtx,
    X509DN::Uptr& derSubjectDN,
    std::map<std::uint32_t, std::string>& x509Extensions,
    std::uint32_t version) {
    CmX509_Ref certSignRequest_ref;
    X509DNRef x509Dn_ref = dynamic_cast<x509::CimpX509DN*>(derSubjectDN.release())->getX509DnRef();
    CipherCtxRef signCtx = dynamic_cast<cryp::CimplSignerPrivateCtx*>(signerCtx.release())->getSignerPrivateCtx();
    if (X509CmClient::Instance().CreateCSR(signCtx, x509Dn_ref, x509Extensions,version, certSignRequest_ref) == 0) {
        CRYP_INFO << "X509CmClient CreateCSR";
        return std::move(std::make_unique<CimpCertSignRequest>(certSignRequest_ref));
    } else {
        CRYP_INFO << "X509CmClient CreateCSR error";
    }
    return CertSignRequest::Uptr();
}

uint32_t CimpX509Provider::ParseOpensslConstraints(uint32_t sslKeyUsage) {
  if (UINT16_MAX == sslKeyUsage) {
    return 0;
  }

  if (0x8000 == sslKeyUsage) {
    return 0x80;
  }

  uint32_t ret = (sslKeyUsage << 8);
  return ret;
}

bool CimpX509Provider::ImportCert(const Certificate::Uptr &cert, const std::string destCertPath) {
    bool ret = false;
    CimpCertificate* cimp_cert = dynamic_cast<CimpCertificate*>(cert.get());
    CryptoCertRef ref = cimp_cert->GetCertRef();
    ret = X509CmClient::Instance().ImportCert(ref,destCertPath);
    return ret;
}

bool CimpX509Provider::ImportCrl(const std::string crlPath) {
    return true;
}

bool CimpX509Provider::SetAsRootOfTrust(const Certificate::Uptr &caCert) {
    //TODO
    return true;
}

Certificate::Status CimpX509Provider::VerifyCert(Certificate::Uptr &cert,
                        const std::string rootCertPath) {
    CimpCertificate* cimp_cert = dynamic_cast<CimpCertificate*>(cert.get());
    CryptoCertRef ref = cimp_cert->GetCertRef();
    auto ret = X509CmClient::Instance().VerifyCert(ref,rootCertPath);
    return static_cast<Certificate::Status>(ret);
}

OcspRequest::Uptr CimpX509Provider::CreateOcspRequest(const std::string certPath, const std::string rootPath) {
    //TODO
    OcspRequest::Uptr uptrcOcsp;
    return uptrcOcsp;
}

Certificate::Status CimpX509Provider::CheckCertStatus(const std::string certPath,
                        const std::string ocspResponsePath,
                        const std::string rootPath) {
    //TODO
    return Certificate::Status::kUnknown;
}

OcspResponse::Uptr CimpX509Provider::ParseOcspResponse(const std::string ocspResponsePath) {
    //TODO
    OcspResponse::Uptr uptrOcsp;
    return uptrOcsp;
}

Certificate::Uptr CimpX509Provider::FindCertByDn(const X509DN &subjectDn,
        const X509DN &issuerDn, time_t validityTimePoint) noexcept {
    CryptoCertRef cert_ref;
    auto ret = X509CmClient::Instance().FindCertByDn(subjectDn, issuerDn,validityTimePoint,cert_ref);
    Certificate::Uptr cimp_cert( new CimpCertificate(cert_ref));
    return cimp_cert;
}


X509* CimpX509Provider::LoadX509Cert(const std::string certPath) {
  BIO* tbio = nullptr;
  cert = nullptr;

  auto PTR_FREE = [&]() -> void {
        if (tbio != nullptr) {
            // 释放 BIO
            BIO_free(tbio);
        }
        if (cert != nullptr) {
            // 释放 X509 CRL
            X509_free(cert);
        }
    };

  try {
      // 从文件创建BIO
      tbio = BIO_new_file(certPath.c_str(), "r");
      if (tbio == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "BIO_new_file fail: " << certPath;
        return nullptr;
      }
      // 从BIO创建x509
      cert = PEM_read_bio_X509(tbio, nullptr, 0, nullptr);
      if (cert == nullptr) {
          PTR_FREE();
          CRYP_ERROR << "PEM_read_bio_X509 fail: " << certPath;
          return nullptr;
      }
    } catch (const std::exception& e) {
          // todo: 处理异常
          PTR_FREE();
          CRYP_ERROR << "ParseCert fail: " << certPath;
          return nullptr;
    }
    return cert;
}

X509_CRL* CimpX509Provider::LoadX509Crl(const std::string crlPath) {
    BIO* tbio = nullptr;
    crl = nullptr;

    auto PTR_FREE = [&]() -> void {
        if (tbio != nullptr) {
            // 释放 BIO
            BIO_free(tbio);
        }
        if (crl != nullptr) {
            // 释放 X509 CRL
            X509_CRL_free(crl);
        }
    };

    try {
      // 从文件创建BIO
      tbio = BIO_new_file(crlPath.c_str(), "r");
      if (tbio == nullptr) {
          PTR_FREE();
          CRYP_ERROR << "BIO_new_file fail: " << crlPath;
          return nullptr;
      }
      // 从BIO创建x509 CRL
      crl = PEM_read_bio_X509_CRL(tbio, nullptr, 0, nullptr);
      if (crl == nullptr) {
          PTR_FREE();
          CRYP_ERROR << "PEM_read_bio_X509_CRL fail: " << crlPath;
          return nullptr;
      }
    } catch (const std::exception& e) {
          // todo: 处理异常
          PTR_FREE();
          CRYP_ERROR << "ParseCert fail: " << crlPath;
          return nullptr;
    }
    return crl;
}

OCSP_RESPONSE* CimpX509Provider::LoadOcspResponse(const std::string ocspResponsePath) {
    BIO* tbio = nullptr;
    ocspresp = nullptr;

    auto PTR_FREE = [&]() -> void {
        if (tbio != nullptr) {
            // 释放 BIO
            BIO_free(tbio);
        }
        if (ocspresp != nullptr) {
            // 释放 OCSP RESPONSE
            OCSP_RESPONSE_free(ocspresp);
        }
    };

    try {
        // 从文件创建BIO
        tbio = BIO_new_file(ocspResponsePath.c_str(), "r");
        if (tbio == nullptr) {
            PTR_FREE();
            CRYP_ERROR << "BIO_new_file fail: " << ocspResponsePath;
            return nullptr;
        }
        // 从BIO创建OCSP response
        ocspresp = d2i_OCSP_RESPONSE_bio(tbio, nullptr);
        if (ocspresp == nullptr) {
            PTR_FREE();
            CRYP_ERROR << "d2i_OCSP_RESPONSE_bio fail: " << ocspResponsePath;
            return nullptr;
        }
      } catch (const std::exception& e) {
            // todo: 处理异常
            PTR_FREE();
            CRYP_ERROR << "d2i_OCSP_RESPONSE_bio fail ocur exception" << ocspResponsePath;
            return nullptr;
      }
      return ocspresp;
}

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
