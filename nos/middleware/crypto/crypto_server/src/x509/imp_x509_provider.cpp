#include "x509/imp_x509_provider.h"

#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/ocsp.h>
#include <dirent.h>

#include "common/crypto_logger.hpp"
#include "x509/x509_provider.h"
#include "x509/imp_certificate.h"
#include "x509/imp_x509_dn.h"
#include "server/crypto_server_config.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

X509* cert = nullptr;  // for LoadX509Cert from file path
X509_CRL* crl = nullptr;  // for LoadX509Crl from file path
OCSP_RESPONSE* ocspresp = nullptr;  // for Load ocsp response from file path

// X509Provider* ImpX509Provider::getInstance() {
//     static ImpX509Provider instance;
//     return &instance;
// }

void ImpX509Provider::Init() {
    CryptoLogger::GetInstance().InitLogging("crypto","crypto service",
        CryptoLogger::CryptoLogLevelType::CRYPTO_INFO, //the log level of application
        hozon::netaos::log::HZ_LOG2FILE, //the output log mode
        "/opt/usr/log/soc_log/", //the log file directory, active when output log to file
        10, //the max number log file , active when output log to file
        20 //the max size of each  log file , active when output log to file
    );
    CryptoLogger::GetInstance().CreateLogger("CRPS");
    CRYP_DEBUG << "X509Provider::Init";
}

void ImpX509Provider::DeInit() {
  CRYP_DEBUG << "X509Provider::DeInit";
//   if (nullptr != instance_) {
//     delete instance_;
//     instance_ = nullptr;
//     CRYP_INFO << "delete X509Provider instance_";
//   }
}

ImpX509Provider::~ImpX509Provider() {
}

X509* ImpX509Provider::LoadX509Cert(const std::string certPath) {
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

X509_CRL* ImpX509Provider::LoadX509Crl(const std::string crlPath) {
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

OCSP_RESPONSE* ImpX509Provider::LoadOcspResponse(const std::string ocspResponsePath) {
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

X509DN::Uptr ImpX509Provider::BuildDn(std::string dn) noexcept {
    X509DN::Uptr uptr(new ImpX509DN());
    if (uptr->SetDn(dn)) {
        return uptr;
    }
    return X509DN::Uptr();
}

Certificate::Uptr ImpX509Provider::ParseCert(ReadOnlyMemRegion certMem, Serializable::FormatId formatId) {
    ImpCertificate* impl_cert = new ImpCertificate();
    uint32_t format = (formatId == Serializable::kFormatDerEncoded) ? Serializable::kFormatDerEncoded : Serializable::kFormatPemEncoded;
    X509* x509cert__ = nullptr;
    if (Serializable::kFormatDerEncoded == format) {
        const unsigned char *p = reinterpret_cast<const unsigned char *>(certMem.data());
        x509cert__ = d2i_X509(nullptr, &p, sizeof(certMem.size()));
        impl_cert->cert_ = x509cert__;
        impl_cert->formatId = Serializable::kFormatDerEncoded;
    } else if (Serializable::kFormatPemEncoded == format) {
        std::string certStr(reinterpret_cast<const char*>(certMem.data()), certMem.size());
        BIO* bio = BIO_new_mem_buf(reinterpret_cast<const char*>(certStr.data()), certStr.size());
        x509cert__ = PEM_read_bio_X509(bio, NULL, NULL, NULL);
        impl_cert->cert_ = x509cert__;
        impl_cert->formatId = Serializable::kFormatPemEncoded;
    } else {
        CRYP_ERROR <<"ParseCert ,formate: "<< static_cast<int>(formatId)<<"is not suposed !";
    }

    if (!x509cert__) {
        return Certificate::Uptr();
    }
    return std::unique_ptr<ImpCertificate>(impl_cert);
}

// todo: 依赖cryp做签名和加密
// CertSignRequest::Uptr
// X509Provider::CreateCertSignRequest(cryp::SignerPrivateCtx::Uptr signerCtx,
// std::string& derSubjectDN, std::string& x509Extensions, std::uint32_t
// version) {
//     CertSignRequest::Uptr uptrcCsr(new CertSignRequest());
//     uptrcCsr->SubjectDn().SetDn(derSubjectDN);
//     uptrcCsr->x509Extensions = x509Extensions;
//     uptrcCsr->version = version;
//     uptrcCsr->signature = signerCtx;
//     return CertSignRequest::Uptr();
// }

CertSignRequest::Uptr ImpX509Provider::CreateCertSignRequest(
    cryp::SignerPrivateCtx::Uptr& signerCtx,

    X509DN::Uptr& derSubjectDN,
    std::map<std::uint32_t, std::string>& x509Extensions,
    std::uint32_t version) {

  CertSignRequest::Uptr uptrcCsr(new ImpCertSignRequest());
  uptrcCsr->pSubjectDN = std::move(derSubjectDN);
  uptrcCsr->x509Extensions = x509Extensions;
  uptrcCsr->version = version;
  uptrcCsr->signerCtx = std::move(signerCtx);
  return uptrcCsr;
}

uint32_t ImpX509Provider::ParseOpensslConstraints(uint32_t sslKeyUsage) {
  if (UINT16_MAX == sslKeyUsage) {
    return 0;
  }

  if (0x8000 == sslKeyUsage) {
    return 0x80;
  }

  uint32_t ret = (sslKeyUsage << 8);
  return ret;
}

bool ImpX509Provider::ImportCert(const Certificate::Uptr &cert, const std::string destCertPath) {
    FILE* file = nullptr;
    int ret;
    ImpCertificate* imp_cert = dynamic_cast<ImpCertificate*>(cert.get());
    if (cert->formatId == Serializable::kFormatPemEncoded) {
        file = fopen(destCertPath.c_str(), "w");
        ret = PEM_write_X509(file, imp_cert->cert_);
        CRYP_INFO <<"ImportCert kFormatPemEncoded :" << destCertPath.c_str() <<" ret :"<< ret;
    } else if (imp_cert->formatId == Serializable::kFormatDerEncoded) {
        file = fopen(destCertPath.c_str(), "r");
        ret = i2d_X509_fp(file, imp_cert->cert_);
        CRYP_INFO <<"ImportCert kFormatDerEncoded :" << destCertPath.c_str() <<" ret :"<< ret;
    } else {
        CRYP_ERROR<<"ImportCert cert failed, formate is :"<< static_cast<int>(imp_cert->formatId);
    }

    fclose(file);
    if (imp_cert->IsCa() && imp_cert->IsRoot()) {
        CRYP_INFO << "ImportCert: this cert is root cert !";
        auto set_root_res = SetAsRootOfTrust(cert);
        if (!set_root_res) {
            CRYP_ERROR << "ImportCert: set cert as root failed";
            return false;
        }
    } else {
        CRYP_INFO << "ImportCert: this cert is not root cert !";
    }
    return true;
}

bool ImpX509Provider::ImportCrl(const std::string crlPath) {
    X509_CRL* crl = nullptr;
    X509* rootCert = nullptr;
    EVP_PKEY* pbKey = nullptr;

    auto PTR_FREE = [&]() -> void {
        if (rootCert != nullptr) {
            // 释放 X509
            X509_free(rootCert);
        }
        if (pbKey != nullptr) {
            // 释放 X509
            EVP_PKEY_free(pbKey);
        }
        if (crl != nullptr) {
            // 释放 X509 CRL
            X509_CRL_free(crl);
        }
    };

    crl = LoadX509Crl(crlPath);
    if (crl == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "LoadX509Crl failed, crl is null";
        return false;
    }

    rootCert = LoadX509Cert(ROOT_CERT_STORAGE_PATH);
    if (rootCert == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "LoadX509Crl failed, rootCert is null";
        return false;
    }
    // 提取证书公钥
    pbKey = X509_get_pubkey(rootCert);
    if (pbKey) {
        // // 验证 CRL
        if (X509_CRL_verify(crl, pbKey) <= 0) {
            PTR_FREE();
            CRYP_ERROR << "X509_CRL_verify fail: crl is invalid" << crlPath;
            return false;
        }
    } else {
        PTR_FREE();
        CRYP_ERROR << "X509_get_pubkey fail, import crl failed" << crlPath;
        return false;
    }

    FILE* fp = fopen(CERT_CRL_STORAGE_PATH, "w");
    if (fp == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "Open destCertPath fail: " << CERT_CRL_STORAGE_PATH;
        return false;
    } else {
          // 把CRL以PEM格式存入文件
          PEM_write_X509_CRL(fp, crl);
    }
    PTR_FREE();
    fclose(fp);
    return true;
}

bool ImpX509Provider::SetAsRootOfTrust(const Certificate::Uptr &caCert) {
    // Import ca certificate into x509 provider.
    FILE* file = fopen(ROOT_CERT_STORAGE_PATH, "w");
    ImpCertificate* imp_cert = dynamic_cast<ImpCertificate*>(caCert.get());
    if (imp_cert->formatId == Serializable::kFormatPemEncoded) {
        if (0 <= PEM_write_X509(file, imp_cert->cert_)) {
            CRYP_INFO <<"SetAsRootOfTrust success !";
        } else {
             CRYP_ERROR<<"SetAsRootOfTrust PEM_write_X509 failed!";
             return false;
        }
        fclose(file);
    } else {
        CRYP_ERROR<<"SetAsRootOfTrust failed !";
        return false;
    }
    return true;
}

Certificate::Status ImpX509Provider::VerifyCert(Certificate::Uptr &cert,
                        const std::string rootCertPath) {
    X509_STORE* store = nullptr;
    X509_STORE_CTX *ctx = nullptr;
    X509* rootCert = nullptr;
    X509* signedCert = nullptr;
    X509_CRL* crl = nullptr;

    auto PTR_FREE = [&]() -> void {
        if (store != nullptr) {
            // 释放证书存储区
            X509_STORE_free(store);
        }
        if (ctx != nullptr) {
            // 释放证书存储区上下文环境
            X509_STORE_CTX_free(ctx);
        }
        if (rootCert != nullptr) {
            // 释放 X509
            X509_free(rootCert);
        }
        if (signedCert != nullptr) {
            // 释放 X509
            X509_free(signedCert);
        }
        if (crl != nullptr) {
            // 释放 X509 CRL
            X509_CRL_free(crl);
        }
    };
    // 创建证书存储区
    store = X509_STORE_new();
    if (store == nullptr) {
        CRYP_ERROR << "X509_STORE_new fail";
        PTR_FREE();
        return Certificate::Status::kUnknown;
    }
    // 创建证书存储区上下文环境函数
    ctx = X509_STORE_CTX_new();
    if (ctx == nullptr) {
        CRYP_ERROR << "X509_STORE_new fail";
        PTR_FREE();
        return Certificate::Status::kUnknown;
    }

    // load root cert
    rootCert = LoadX509Cert(rootCertPath);
    if (rootCert == nullptr) {
        CRYP_ERROR << "ConstructX509Cert root cert fail";
        PTR_FREE();
        return Certificate::Status::kNoTrust;
    }

    // load siged cert
    // signedCert = LoadX509Cert(signedCertPath);
    // if (signedCert == nullptr) {
    //     CRYP_ERROR << "ConstructX509Cert root cert fail";
    //     PTR_FREE();
    //     return Certificate::Status::kInvalid;
    // }

    signedCert = dynamic_cast<ImpCertificate*>(cert.get())->cert_;
    // 向证书存储区添加证书
    if (!X509_STORE_add_cert(store, rootCert)) {
        CRYP_ERROR << "X509_STORE_add_cert fail";
        PTR_FREE();
        return Certificate::Status::kNoTrust;
    }

    // 初始化证书存储区上下文环境函数
    if (!X509_STORE_CTX_init(ctx, store, signedCert, nullptr)) {
        CRYP_ERROR << "X509_STORE_CTX_init fail";
        PTR_FREE();
        return Certificate::Status::kUnknown;
    }
    // 验证证书函数
    if (X509_verify_cert(ctx) <= 0) {
        CRYP_ERROR << "X509_verify_cert fail";
        PTR_FREE();
        return Certificate::Status::kInvalid;
    }

    // load crl cert
    crl = LoadX509Crl(CERT_CRL_STORAGE_PATH);
    if (crl == nullptr) {
        CRYP_ERROR << "LoadX509Crl fail";
        PTR_FREE();
        return Certificate::Status::kUnknown;
    }

    int is_revoked =X509_CRL_get0_by_cert(crl, nullptr, signedCert);
    if (is_revoked == 1) {
        CRYP_ERROR << "Certificate is revoked";
        PTR_FREE();
        return Certificate::Status::kExpired;
    } else {
        CRYP_INFO << "VerifyCert Certificate success!";
    }

    PTR_FREE();
    return Certificate::Status::kValid;
}

OcspRequest::Uptr ImpX509Provider::CreateOcspRequest(const std::string certPath, const std::string rootPath) {
    OcspRequest::Uptr uptrcOcsp(new ImpOcspRequest(certPath, rootPath));
    return uptrcOcsp;
}

Certificate::Status ImpX509Provider::CheckCertStatus(const std::string certPath,
                        const std::string ocspResponsePath,
                        const std::string rootPath) {
    OCSP_RESPONSE* resp = nullptr;
    OCSP_BASICRESP* basicresp = nullptr;
    X509_STORE* store = nullptr;
    X509* cert = nullptr;
    X509* root = nullptr;
    OCSP_CERTID* id = nullptr;

    auto PTR_FREE = [&]() -> void {
        if (resp != nullptr) {
            // 释放 OCSP RESPONSE
            OCSP_RESPONSE_free(resp);
        }
        if (cert != nullptr) {
            // 释放 X509
            X509_free(cert);
        }
        if (root != nullptr) {
            // 释放 X509
            X509_free(root);
        }
        if (basicresp != nullptr) {
            // 释放 OCSP BASICRESP
            OCSP_BASICRESP_free(basicresp);
        }
        if (store != nullptr) {
            X509_STORE_free(store);
        }
        if (id != nullptr) {
            OCSP_CERTID_free(id);
        }
    };
    resp = LoadOcspResponse(ocspResponsePath);
    if (resp == nullptr) {
        CRYP_ERROR << "LoadOcspResponse fail" << ocspResponsePath;
        return Certificate::Status::kUnknown;
    }

    int respStatus = OCSP_response_status(resp);
    if (respStatus != OCSP_RESPONSE_STATUS_SUCCESSFUL) {
        PTR_FREE();
        CRYP_ERROR << "OCSP_response_status fail" << respStatus;
        return Certificate::Status::kUnknown;
    }
    basicresp = OCSP_response_get1_basic(resp);
    if (basicresp == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "OCSP_response_get1_basic fail" << ocspResponsePath;
        return Certificate::Status::kUnknown;
    }

    store = X509_STORE_new();
    if (store == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "X509_STORE_new fail" << ocspResponsePath;
        return Certificate::Status::kUnknown;
    }

    cert = LoadX509Cert(certPath);
    root = LoadX509Cert(rootPath);

    STACK_OF(X509)* certs = sk_X509_new_null();
    if (!sk_X509_push(certs, cert)) {
        PTR_FREE();
        CRYP_ERROR << "sk_X509_push fail" << ocspResponsePath;
        return Certificate::Status::kUnknown;
    }
    if (!sk_X509_push(certs, root)) {
        PTR_FREE();
        CRYP_ERROR << "sk_X509_push fail" << ocspResponsePath;
        return Certificate::Status::kUnknown;
    }
    X509_STORE_add_cert(store, cert);
    X509_STORE_add_cert(store, root);
    // TO DO test
    // 检查基本响应消息是否已正确签名，以及签名者证书是否可以验证,OCSP_basic_verify 一直循环直到找到根 CA
    /*
    if (OCSP_basic_verify(basicresp, certs, store, 0) <= 0) {
        PTR_FREE();
        CRYP_ERROR << "OCSP_basic_verify fail" << ocspResponsePath;
        return Certificate::Status::kUnknown;
    }
    */
    // 创建并返回一个新的OCSP_CERTID结构
    id = OCSP_cert_to_id(nullptr, cert, root);
    int status, reason;
    //  获取单个证书状态，返回值为其状态
    if ( OCSP_resp_find_status(basicresp, id, &status, &reason, nullptr, nullptr, nullptr) <= 0 ) {
        PTR_FREE();
        CRYP_ERROR << "OCSP_resp_find_status fail" << ocspResponsePath;
        return Certificate::Status::kUnknown;
    }
    switch (status) {
    case V_OCSP_CERTSTATUS_GOOD:
      return Certificate::Status::kValid;
    case V_OCSP_CERTSTATUS_REVOKED:
      return Certificate::Status::kExpired;
    default:
      return Certificate::Status::kUnknown;
    }
}

OcspResponse::Uptr ImpX509Provider::ParseOcspResponse(const std::string ocspResponsePath) {
    OCSP_RESPONSE* resp = nullptr;

    auto PTR_FREE = [&]() -> void {
        if (resp != nullptr) {
            // 释放 OCSP RESPONSE
            OCSP_RESPONSE_free(resp);
        }
    };
    resp = LoadOcspResponse(ocspResponsePath);
    if (resp == nullptr) {
        PTR_FREE();
        CRYP_ERROR << "LoadOcspResponse fail" << ocspResponsePath;
        return nullptr;
    }
    int respStatus = OCSP_response_status(resp);
    OcspResponse::Uptr uptrOcsp(new ImpOcspResponse(respStatus));
    return uptrOcsp;
}

Certificate::Uptr ImpX509Provider::FindCertByDn(const X509DN &subjectDn,
        const X509DN &issuerDn, time_t validityTimePoint) noexcept {
    ImpCertificate* impl_cert = new ImpCertificate();

    std::string path =  CryptoConfig::Instance().GetDeviceCertPath();
    CRYP_INFO << "cert_storage_path:"<< path;
    if (path.c_str() == nullptr) {
        impl_cert->cert_ = nullptr;
        CRYP_ERROR << "FindCertByDn file path is null !" << path;
        return std::unique_ptr<Certificate>(impl_cert);
    }
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        impl_cert->cert_ = nullptr;
        CRYP_ERROR << "FindCertByDn failed to open directory " << path;
        return std::unique_ptr<Certificate>(impl_cert);
    }
    std::string subject_dn_str = const_cast<X509DN&>(subjectDn).GetDnString(); // 指定subject证书的X509Dn字符串
    std::string issuer_dn_str= const_cast<X509DN&>(issuerDn).GetDnString();; // 指定issuer证书的X509Dn字符串
    CRYP_INFO << "FindCertByDn sourch cert subject_dn_str: " << subject_dn_str.c_str();
    CRYP_INFO << "FindCertByDn sourch cert issuer_dn_str: " << issuer_dn_str.c_str();

    X509* curr_cert = NULL;
    X509* founder_cert = NULL;
    // 遍历目录下的所有文件
    struct dirent* ent = NULL;
    while ((ent = readdir(dir))) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        std::string full_path = path + ent->d_name;

        if (ent->d_type == DT_REG) {
            FILE* fp = fopen(full_path.c_str(), "r");
            curr_cert = PEM_read_X509(fp, NULL, NULL, NULL);
            fclose(fp);

            if (!curr_cert) {
                fp = fopen(full_path.c_str(), "r");
                curr_cert = d2i_X509_fp(fp, NULL);
                fclose(fp);
            }
            if (curr_cert){
                // 将 Subject 字段中的属性值打印出来
                X509_NAME* subject_dn = X509_get_subject_name(curr_cert);
                char subject_buf[4096];
                X509_NAME_oneline(subject_dn, subject_buf, sizeof(subject_buf));
                CRYP_INFO  << "FindCertByDn curr_cert path: " << full_path.c_str();
                CRYP_INFO <<"FindCertByDn curr_cert subject_dn: " << subject_buf;
                std::string str_subject(subject_buf);

                // 将 Issuer 字段中的属性值打印出来
                X509_NAME* issuer_dn = X509_get_issuer_name(curr_cert);
                char issuer_buf[4096];
                X509_NAME_oneline(issuer_dn, issuer_buf, sizeof(issuer_buf));
                CRYP_INFO  << "FindCertByDn curr_cert issuer_dn: " << issuer_buf;
                std::string str_issuer(issuer_buf);
                if (0 == subject_dn_str.compare(str_subject) && 0 == issuer_dn_str.compare(str_issuer)) {
                    CRYP_INFO  << "subject_dn issuer_dn compared pass !";
                    struct tm notAfter;
                    ASN1_TIME_to_tm(X509_getm_notAfter(curr_cert), &notAfter);
                    struct tm notBefore;
                    ASN1_TIME_to_tm(X509_getm_notBefore(curr_cert), &notBefore);
                    time_t notAfterTime = mktime(&notAfter);
                    time_t notBeforeTime = mktime(&notBefore);
                    CRYP_INFO << "FindCertByDn found cert: validityTimePoint:"<< validityTimePoint << " notBeforeTime"<<notBeforeTime<<" notAfterTime: "<<notAfterTime ;
                    if (((validityTimePoint - notBeforeTime) < 60)
                                && validityTimePoint < notAfterTime) {
                        founder_cert = curr_cert;
                        CRYP_INFO  << "notBeforeTime notAfterTime check pass !";
                        break;
                    } else {
                        CRYP_WARN << "FindCertByDn found InvalidityTime cert, pelease check cert time !" ;
                    }
                }
            } else {
                CRYP_WARN << "FindCertByDn can not open file: " << full_path;
            }
        }
    }

    closedir(dir);
    impl_cert->cert_ = founder_cert;
    return std::unique_ptr<Certificate>(impl_cert);
}

}  // namespace x509
}  // namespace crypto
}  // namespace netaos
}  // namespace hozon
