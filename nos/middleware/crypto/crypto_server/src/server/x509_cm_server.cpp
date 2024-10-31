#include "crypto_cm_server.h"
#include <mutex>
#include <memory>
#include "cm/include/method.h"
#include "common/crypto_error_domain.h"
#include "common/inner_types.h"
#include "common/crypto_logger.hpp"
#include "common/type_converter.h"
#include "server/resource_keeper.h"
#include "cryp/symmetric_block_cipher_ctx.h"
#include "idl/generated/crypto.h"
#include "idl/generated/cryptoPubSubTypes.h"
#include "keys/key_slot_prototype_props.h"
#include "keys/json_parser.h"

#include "keys/imp_keyslot.h"
#include "keys/imp_key_storage_provider.h"
#include <openssl/pkcs12.h>
#include "cryp/cryobj/imp_private_key.h"
#include "keys/imp_keyslot.h"
#include "x509_cm_server.h"
#include "x509/imp_cert_sign_request.h"
#include "server/resource_keeper.h"
#include "cryp/imp_signer_private_ctx.h"

namespace hozon {
namespace netaos {
namespace crypto {

const uint32_t CRYPTO_DOMAIN = 2;
static X509CmServer* sinstance_ = nullptr;
static std::mutex sinstance_mutex_;

X509CmServer& X509CmServer::Instance()  {

    std::lock_guard<std::mutex> lock(sinstance_mutex_);
    if (!sinstance_) {
        sinstance_ = new X509CmServer();
    }

    return *sinstance_;
}

void X509CmServer::Destroy()  {

    std::lock_guard<std::mutex> lock(sinstance_mutex_);
    if (sinstance_) {
        delete sinstance_;
    }
}

bool X509CmServer::Init()  {
    std::call_once(onceFlag, [&](){
        x509_provider_ = hozon::netaos::crypto::LoadX509Provider();
    });
    return true;
}

void X509CmServer::Deinit()  {
     if (x509_provider_) {
        x509_provider_.reset(nullptr); 
    }
}

void X509CmServer::Stop() {
    stopped_ = true;
    parse_cert_method_server_.Stop();
    create_cert_sign_method_server_.Stop();
    exportASN1CertSignMethodServer_.Stop();
    x509BuildOnMethodServer_.Stop();
    setDnMethodServer_.Stop();
}

void X509CmServer::Start() {
    parse_cert_method_server_.Start(CRYPTO_DOMAIN, "ParseCertRequest");
    create_cert_sign_method_server_.Start(CRYPTO_DOMAIN, "CreateCertSignRequest");
    exportASN1CertSignMethodServer_.Start(CRYPTO_DOMAIN, "ExportASN1CertSignRequest");
    x509BuildOnMethodServer_.Start(CRYPTO_DOMAIN, "BuildDnRequest");
    setDnMethodServer_.Start(CRYPTO_DOMAIN, "SetDnRequest");
}

int32_t X509CmServer::ParseCert(const std::shared_ptr<ParseCertRequest> req, std::shared_ptr<ParseCertResult> resp) {
    // ReadOnlyMemRegion cert_mem(req->certMem().data(),req->certMem().size());
    // Serializable::FormatId format_id = req->formatId();
    // auto result = CryptoServer::Instance().x509_provider_->ParseCert(cert_mem,format_id);
    // if (!result) {
    //     CRYP_INFO << "Server x509_provider_->ParseCert failed.";
    //     return -1;
    // } else {
    //     CRYP_INFO << "x509_provider_->ParseCert success.";
    //     x509::Certificate* raw_ptr = std::move(result).release();
    //     ResourceKeeper::Instance().KeepPrivateKey(raw_ptr);

    //     // Populate key ref as output.
    //     key_ref = GetCryptoKeyRefInfo(raw_ptr);

    //     res = CryptoErrc::kSuccess;
    // }
    return 0;
}

int32_t X509CmServer::CreateCertSign(const std::shared_ptr<CreateCertSignRequest> req, std::shared_ptr<CreateCSRResult> resp) {
    CryptoErrc res = CryptoErrc::kUnsupported;

    if (req->key().ref() == 0) {
        CRYP_ERROR << "CreateCertSign erro ,req is null 2";
        return static_cast<int32_t>(res);
    }
    x509::X509DN* dn = ResourceKeeper::Instance().QueryX509Dn(req->x509dn_ref().ref());

    x509::CertSignRequest::Uptr uptrcCsr(new  x509::ImpCertSignRequest());
    if (dn) {
        CRYP_INFO << "CreateCertSign dn :" << dn->GetDnString();
        uptrcCsr->pSubjectDN.reset(dn);
        CRYP_INFO << "after CreateCertSign dn :" << uptrcCsr->pSubjectDN->GetDnString();
    }
    uptrcCsr->x509Extensions = req->x509Extensions();
    uptrcCsr->version = req->version();

    cryp::ImpSignerPrivateCtx* impSignerPrivateCtx = dynamic_cast<cryp::ImpSignerPrivateCtx*>(ResourceKeeper::Instance().QuerySignerPrivateCtx(req->key().ref()));
    // uptrcCsr->signerCtx = std::move(dynamic_cast<cryp::SignerPrivateCtx::Uptr&>(*impSignerPrivateCtx));
    uptrcCsr->signerCtx.reset(impSignerPrivateCtx);
    x509::CertSignRequest* raw_ptr = std::move(uptrcCsr).release();
    if (!raw_ptr) {
        CRYP_ERROR << "CertSignRequest raw_ptr is null ";
        return static_cast<int32_t>(res);
    }
    ResourceKeeper::Instance().KeepCertSignRequest(const_cast<x509::CertSignRequest *>(raw_ptr));
    resp->certSignRequest_ref().ref() = reinterpret_cast<uint64_t>(raw_ptr);
    res = CryptoErrc::kSuccess;
    return static_cast<int32_t>(res);
}

int32_t X509CmServer::ExportASN1CSR(const std::shared_ptr<ExportASN1CertSignRequest> req, std::shared_ptr<ExportASN1CertSignResult> resp) {
    CryptoErrc res = CryptoErrc::kUnsupported;
    x509::CertSignRequest* csr_req = ResourceKeeper::Instance().QueryCertSignRequest(req->certSignRequest_ref().ref());
    std::vector<uint8_t> csr_vec;
    if (csr_req) {
        csr_vec = csr_req->ExportASN1CertSignRequest();
    }
    if (!csr_vec.empty()) {
        res = CryptoErrc::kSuccess;
        resp->signature() = csr_vec;
        resp->code() = static_cast<int>(CryptoErrc::kSuccess);
    } else {
        CRYP_ERROR << "X509CmServer ExportASN1CSR is nullptr";
    }
    return static_cast<int32_t>(res);
}

int32_t X509CmServer::BuildOnDn(const std::shared_ptr<BuildDnRequest> req, std::shared_ptr<BuildDnResult> resp) {
    CryptoErrc res = CryptoErrc::kUnsupported;
    if (!x509_provider_) {
        CRYP_ERROR << "x509_provider_ is null ";
        return static_cast<int32_t>(res);
    }

    auto obj_x509Dn = x509_provider_->BuildDn(req->dn());
    x509::X509DN* raw_ptr = std::move(obj_x509Dn).release();
    if (!raw_ptr) {
        CRYP_ERROR << "x509BuildOn raw_ptr is null ";
        return static_cast<int32_t>(res);
    }
    ResourceKeeper::Instance().KeepX509DN(const_cast<x509::X509DN *>(raw_ptr));
    resp->x509dn_ref().ref() = reinterpret_cast<uint64_t>(raw_ptr);
    res = CryptoErrc::kSuccess;

    return static_cast<int32_t>(res);
}

int32_t X509CmServer::SetDn(const std::shared_ptr<SetDnRequest> req, std::shared_ptr<SetDnResult> resp) {
    CryptoErrc res = CryptoErrc::kUnsupported;
    x509::X509DN* x509_dn = ResourceKeeper::Instance().QueryX509Dn(req->x509dn_ref().ref());
    if (!x509_dn) {
        CRYP_ERROR << "x509_dn is null ";
        return static_cast<int32_t>(res);
    }
     if (req->dn().empty()) {
        CRYP_ERROR << "req dn is null ";
        return static_cast<int32_t>(res);
    }
    if (x509_dn->SetDn(req->dn())) {
        res = CryptoErrc::kSuccess;
        resp->Result() = true;
    } else {
        CRYP_ERROR << "SetDn is error ";
        resp->Result() = false;
    }
    return static_cast<int32_t>(res);
}

X509CmServer::~X509CmServer() {

}

CryptoKeyRef X509CmServer::GetCryptoKeyRefInfo(const cryp::RestrictedUseObject* key) {
    CryptoKeyRef key_ref;
    key_ref.ref = reinterpret_cast<uint64_t>(key);
    key_ref.alg_id = key->GetCryptoPrimitiveId()->GetPrimitiveId();

    key_ref.primitive_id_info.alg_id = key->GetCryptoPrimitiveId()->GetPrimitiveId();

    key_ref.crypto_object_info.objectUid = key->GetObjectId();
    key_ref.crypto_object_info.dependencyUid = key->HasDependence();
    key_ref.crypto_object_info.payloadSize = key->GetPayloadSize();
    key_ref.crypto_object_info.isExportable = key->IsExportable();
    key_ref.crypto_object_info.isSession = key->IsSession();

    key_ref.allowed_usage = key->GetAllowedUsage();

    return key_ref;
}

CipherCtxRef X509CmServer::GetCipherCtxRef(const cryp::CryptoContext* ctx, CipherContextType ctx_type, CryptoTransform transform) {
    CipherCtxRef ctx_ref;
    ctx_ref.ref = reinterpret_cast<uint64_t>(ctx);
    ctx_ref.alg_id = ctx->GetCryptoPrimitiveId()->GetPrimitiveId();
    ctx_ref.ctx_type = static_cast<uint32_t>(ctx_type);
    ctx_ref.transform = static_cast<uint32_t>(transform);
    ctx_ref.is_initialized = ctx->IsInitialized();

    return ctx_ref;
}

}
}
}