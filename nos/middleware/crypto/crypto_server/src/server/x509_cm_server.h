#pragma once

#include <atomic>
#include <vector>
#include "common/inner_types.h"
#include "cryp/crypto_provider.h"
#include "cryp/cryobj/restricted_use_object.h"
#include "idl/generated/crypto.h"
#include "idl/generated/cryptoPubSubTypes.h"
#include "cm/include/method.h"
#include "crypto_server.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {


class X509CmServer {

public:
 class ParseCertMethodServer : public hozon::netaos::cm::Server<ParseCertRequest, ParseCertResult> {
    public:
     ParseCertMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, X509CmServer* server)
         : Server(req_data, resp_data), x509_server(server) {}
     int32_t Process(const std::shared_ptr<ParseCertRequest> req, std::shared_ptr<ParseCertResult> resp) {
            int32_t res = x509_server->ParseCert(req,resp);
            resp->code() = res;
            return res;
        }
     ~ParseCertMethodServer() {}
    private:
     X509CmServer* x509_server;
 };

  class CreateCertSignMethodServer : public hozon::netaos::cm::Server<CreateCertSignRequest, CreateCSRResult> {
    public:
     CreateCertSignMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, X509CmServer* server)
         : Server(req_data, resp_data), x509_server(server) {}
     int32_t Process(const std::shared_ptr<CreateCertSignRequest> req, std::shared_ptr<CreateCSRResult> resp) {
            int32_t res = x509_server->CreateCertSign(req,resp);
            resp->code() = res;
            return res;
        }
     ~CreateCertSignMethodServer() {}
    private:
     X509CmServer* x509_server;
 };

  class ExportASN1CertSignMethodServer : public hozon::netaos::cm::Server<ExportASN1CertSignRequest, ExportASN1CertSignResult> {
    public:
     ExportASN1CertSignMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, X509CmServer* server)
         : Server(req_data, resp_data), x509_server(server) {}
     int32_t Process(const std::shared_ptr<ExportASN1CertSignRequest> req, std::shared_ptr<ExportASN1CertSignResult> resp) {
            int32_t res = x509_server->ExportASN1CSR(req, resp);
            resp->code() = res;
            return res;
        }
     ~ExportASN1CertSignMethodServer() {}
    private:
     X509CmServer* x509_server;
 };

 class X509BuildOnMethodServer : public hozon::netaos::cm::Server<BuildDnRequest, BuildDnResult> {
    public:
     X509BuildOnMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, X509CmServer* server)
         : Server(req_data, resp_data), x509_server(server) {}
     int32_t Process(const std::shared_ptr<BuildDnRequest> req, std::shared_ptr<BuildDnResult> resp) {
            int32_t res = x509_server->BuildOnDn(req, resp);
            resp->code() = res;
            return res;
        }
     ~X509BuildOnMethodServer() {}
    private:
     X509CmServer* x509_server;
 };

  class SetDnMethodServer : public hozon::netaos::cm::Server<SetDnRequest, SetDnResult> {
    public:
     SetDnMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, X509CmServer* server)
         : Server(req_data, resp_data), x509_server(server) {}
     int32_t Process(const std::shared_ptr<SetDnRequest> req, std::shared_ptr<SetDnResult> resp) {
            int32_t res = x509_server->SetDn(req, resp);
            resp->code() = res;
            return res;
        }
     ~SetDnMethodServer() {}
    private:
     X509CmServer* x509_server;
 };

    static X509CmServer& Instance();
    static void Destroy();

    bool Init();
    void Deinit();
    void Stop();
    void Start();

    // void SetCryptoProvider(cryp::CryptoProvider* provider);

    // int32_t GenerateKey(int32_t key_type, uint64_t alg_id, uint32_t allowed_usage, bool is_session, bool is_exportable, CryptoKeyRef& key_ref);
    // int32_t CreateCipherContext(uint64_t alg_id, int32_t ctx_type, CipherCtxRef& ctx_ref);
    int32_t ParseCert(const std::shared_ptr<ParseCertRequest> req, std::shared_ptr<ParseCertResult> resp);
    int32_t CreateCertSign(const std::shared_ptr<CreateCertSignRequest> req, std::shared_ptr<CreateCSRResult> resp);
    int32_t ExportASN1CSR(const std::shared_ptr<ExportASN1CertSignRequest> req, std::shared_ptr<ExportASN1CertSignResult> resp);
    int32_t BuildOnDn(const std::shared_ptr<BuildDnRequest> req, std::shared_ptr<BuildDnResult> resp);
    int32_t SetDn(const std::shared_ptr<SetDnRequest> req, std::shared_ptr<SetDnResult> resp);

    std::unique_ptr<hozon::netaos::crypto::x509::X509Provider> x509_provider_;

private:
    X509CmServer():
    parse_cert_method_server_(std::make_shared<ParseCertRequestPubSubType>(), std::make_shared<ParseCertResultPubSubType>(), this),
    create_cert_sign_method_server_(std::make_shared<CreateCertSignRequestPubSubType>(), std::make_shared<CreateCSRResultPubSubType>(), this),
    exportASN1CertSignMethodServer_(std::make_shared<ExportASN1CertSignRequestPubSubType>(), std::make_shared<ExportASN1CertSignResultPubSubType>(), this),
    x509BuildOnMethodServer_(std::make_shared<BuildDnRequestPubSubType>(), std::make_shared<BuildDnResultPubSubType>(), this),
    setDnMethodServer_(std::make_shared<SetDnRequestPubSubType>(), std::make_shared<SetDnResultPubSubType>(), this)
    {
        Init();
    }
    ~X509CmServer();
    ParseCertMethodServer parse_cert_method_server_;
    CreateCertSignMethodServer create_cert_sign_method_server_;
    ExportASN1CertSignMethodServer exportASN1CertSignMethodServer_;
    X509BuildOnMethodServer x509BuildOnMethodServer_;
    SetDnMethodServer setDnMethodServer_;

    CryptoKeyRef GetCryptoKeyRefInfo(const cryp::RestrictedUseObject* key);
    CipherCtxRef GetCipherCtxRef(const cryp::CryptoContext* ctx, CipherContextType ctx_type, CryptoTransform transform);
    std::atomic<bool> stopped_{false};
    std::recursive_mutex transaction_mutex_;
    std::once_flag onceFlag;

};

}
}
}