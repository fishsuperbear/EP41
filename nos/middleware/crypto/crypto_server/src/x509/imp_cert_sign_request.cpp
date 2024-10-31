#include "x509/imp_cert_sign_request.h"

#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <fstream>
#include <json/json.h>
#include <openssl/pem.h>
#include <filesystem>

#include "common/crypto_logger.hpp"
#include "cryp/imp_signer_private_ctx.h"
#include "cryp/cryobj/imp_public_key.h"
#include "cryp/cryobj/imp_private_key.h"
#include "x509/x509_provider.h"
#include "x509/x509_dn.h"
#include "x509/imp_x509_provider.h"
#include "server/crypto_server_config.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509 {

using namespace hozon::netaos::crypto::cryp;
extern std::map<X509DN::AttributeId, std::string> AttributeId_NID_MAP;

bool ImpCertSignRequest::Verify() { return false; }

std::vector<uint8_t> ImpCertSignRequest::ExportASN1CertSignRequest() {
    X509_REQ *x509_req = X509_REQ_new();
    BIO *out = nullptr;
    X509_NAME *x509_name = nullptr;
    EVP_PKEY* evp_pkey = nullptr;
    if (this->signerCtx) {
        auto priv_key = dynamic_cast<ImpSignerPrivateCtx&>(*this->signerCtx).GetPrivateKey();
        if (priv_key) {
            evp_pkey = dynamic_cast<cryp::ImpPrivateKey*>(priv_key)->get_pkey();
        }
    }

    auto PTR_FREE = [&]() -> void {
        if (!x509_req) {
            X509_REQ_free(x509_req);
        }
        if (!out) {
            BIO_free_all(out);
        }
        if (!evp_pkey) {
            EVP_PKEY_free(evp_pkey);
        }
    };

    if (!evp_pkey) {
        CRYP_ERROR << "X509_REQ_set_version error";
        PTR_FREE();
        return std::vector<uint8_t>();
    }
    // 1. set version of x509 req
    // csr当前只有一个版本，为了避免错误，不适用外部传入的version
    if(!X509_REQ_set_version(x509_req, X509_REQ_VERSION_1)) {
        CRYP_ERROR << "X509_REQ_set_version error";
        PTR_FREE();
    }

    // 2. set subject dn of x509 req
    x509_name = X509_REQ_get_subject_name(x509_req);
    CRYP_INFO << "pSubjectDN :" <<pSubjectDN->GetDnString();
    pSubjectDN->SetDn(pSubjectDN->GetDnString());
    for (auto it: pSubjectDN->attributeMap) {
        auto tmpIt = AttributeId_NID_MAP.find(it.first);
        if ( tmpIt != AttributeId_NID_MAP.end() ) {
            const char *field = tmpIt->second.c_str();
            if(!X509_NAME_add_entry_by_txt(x509_name, field, MBSTRING_ASC, (const unsigned char *)it.second.c_str(), -1, -1, 0)) {
                CRYP_ERROR << "X509_NAME_add_entry_by_txt error" << " filed: " << field << " value: " << it.second.c_str();
                PTR_FREE();
            }
        }
    }

    if (!X509_REQ_set_pubkey(x509_req, evp_pkey)) {
        CRYP_ERROR << "X509_REQ_set_pubkey error";
        PTR_FREE();
        return std::vector<uint8_t>();
    }

    // 4 set ext info of x509 req
    STACK_OF(X509_EXTENSION) * exts;
    exts = sk_X509_EXTENSION_new_null();
    for (auto it: x509Extensions)
    {
        add_ext(exts, x509_req, it.first, it.second.c_str());
    }
    X509_REQ_add_extensions(x509_req, exts);
    sk_X509_EXTENSION_pop_free(exts, X509_EXTENSION_free);

    if (!X509_REQ_sign(x509_req, evp_pkey, EVP_sha256()))
    {
        CRYP_ERROR << "X509_REQ_sign error";
        PTR_FREE();
        return std::vector<uint8_t>();
    }

    std::vector<uint8_t> pem_vec;
    BIO* bio = BIO_new(BIO_s_mem());
    if (bio) {
        if (PEM_write_bio_X509_REQ(bio, x509_req)) {
            char* buffer = nullptr;
            long length = BIO_get_mem_data(bio, &buffer);
            pem_vec.insert(pem_vec.end(), buffer, buffer + length);
            CRYP_INFO << "PEM_write_bio_X509_REQ successed!";
        } else {
            CRYP_ERROR << "PEM_write_bio_X509_REQ failed!";
        }
        BIO_free(bio);
    }
    return pem_vec;
}

unsigned ImpCertSignRequest::Version() 
{
    return X509_REQ_VERSION_1; 
}

ImpCertSignRequest::ImpCertSignRequest() {
}

ImpCertSignRequest::~ImpCertSignRequest() {}

int ImpCertSignRequest::add_ext(STACK_OF(X509_EXTENSION) * sk, X509_REQ *req, int nid, const char *value)
{
    X509_EXTENSION *ex;
    X509V3_CTX ctx;
    X509V3_set_ctx_nodb(&ctx);
    X509V3_set_ctx(&ctx, NULL, NULL, req, NULL, 0);

    ex = X509V3_EXT_conf_nid(NULL, &ctx, nid, value);
    if (!ex)
    {
        CRYP_ERROR << "X509V3_EXT_conf_nid generated error";
        return 0;
    }
    sk_X509_EXTENSION_push(sk, ex);
    return 1;
}

uint32_t ImpCertSignRequest::GetPathLimit() {
    return 0;
}

bool ImpCertSignRequest::IsCa() {
    return false;
}

const X509DN& ImpCertSignRequest::SubjectDn() {
    return *this->pSubjectDN.get();
}

X509Provider& ImpCertSignRequest::MyProvider() {
    X509Provider* prov = new ImpX509Provider;
    return *prov;
}

}
}
}
}  // namespace hozon