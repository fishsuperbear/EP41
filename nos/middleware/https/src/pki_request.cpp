/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: pki_requet.cpp is designed for https.
 */

#include <dirent.h>
#include <openssl/bio.h>
// #include <openssl/core_names.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "pki_request.h"
#include "json/json.h"
#include "rsa_pss.h"
#include "http_client.h"
#include "log_moudle_init.h"

namespace hozon {
namespace netaos {
namespace https {
using namespace hozon::netaos::crypto;
// const std::uint32_t MAX_SIGN_VALUE_FILE_SIZE = 16 * 1024;
// const std::uint32_t READ_FILE_BUFF_SIZE = 1024;

void PkiRequest::Init() {
    if (http_client_) {
          http_client_->Init();
          http_client_->Start();
    } else {
        HTTPS_ERROR << "http_client_ is null";
    }
}

PkiRequest::PkiRequest(
    std::string url,
    std::string save_file_path,
    std::map<std::string, std::string> headers,
    std::string post_data,
    std::shared_ptr<std::vector<uint8_t>> post_data_v2,
    int sdkType,
    std::string client_ap_priv_key_slot,
    std::string client_priv_key_file,
    std::string client_cert_chain,
    std::string client_key_cert_p12,
    std::string client_key_cert_p12_pass,
    std::string root_ca_cert_file)
    :url_(url),
    save_file_path_(save_file_path),
    headers_(headers),
    post_data_(post_data),
    post_data_v2_(post_data_v2),
    sdkType_(sdkType),
    client_ap_priv_key_slot_(client_ap_priv_key_slot),
    client_priv_key_file_(client_priv_key_file),
    client_cert_chain_(client_cert_chain),
    client_key_cert_p12_(client_key_cert_p12),
    client_key_cert_p12_pass_(client_key_cert_p12_pass),
    root_ca_cert_file_(root_ca_cert_file) {
    if (LogModuleInit::getInstance()) {
        LogModuleInit::getInstance()->initLog();
    }
    http_client_ = std::make_shared<HttpClient>();
}

std::string url;
    std::string save_file_path;
    std::map<std::string, std::string> headers;
    std::string post_data;
    std::shared_ptr<std::vector<uint8_t>> post_data_v2;

    int sdkType;
    std::string client_ap_priv_key_slot;
    std::string client_priv_key_file;
    std::string client_cert_chain;
    std::string client_key_cert_p12;
    std::string client_key_cert_p12_pass;
    std::string root_ca_cert_file;

int PkiRequest::Download(ResponseHandler handler) {
    req_ptr = std::make_shared<hozon::netaos::https::Request>();

    req_ptr->url = url_;
    req_ptr->save_file_path = save_file_path_;
    req_ptr->headers = headers_;
    req_ptr->post_data = post_data_;
    req_ptr->post_data_v2 = post_data_v2_;
    req_ptr->sdkType = sdkType_;
    req_ptr->client_ap_priv_key_slot = client_ap_priv_key_slot_;
    req_ptr->client_priv_key_file = client_priv_key_file_;
    req_ptr->client_cert_chain = client_cert_chain_;
    req_ptr->client_key_cert_p12 = client_key_cert_p12_;
    req_ptr->root_ca_cert_file = root_ca_cert_file_;
    if (http_client_) {
        return http_client_->HttpRequest(req_ptr, handler);
    } else {
        HTTPS_ERROR << "http_client_ is null";
        return -1;
    }
}

bool PkiRequest::CancelDownLoad() {
    // if (http_client_) {
    //     http_client_->CancelRequest();
    //     return true;
    // } else {
    //     HTTPS_ERROR << "http_client_ is null";
    //     return false;
    // }
}

PkiRequest::~PkiRequest() {
}

// hozon::netaos::crypto::x509::Certificate::Uptrc PkiRequest::FindIssuerCert(const hozon::netaos::crypto::x509::X509DN& subject_dn, const hozon::netaos::crypto::x509::X509DN& issuer_dn) {
//     if (http_client_) {
//         return http_client_->FindIssuerCert(subject_dn, issuer_dn);
//     } else {
//         HTTPS_ERROR << "http_client_ is null";
//         return x509::Certificate::Uptrc();
//     }
// }

// DataBuffer PkiRequest::CreateClientCsrKeyPair(const std::string& priv_key_slot_uuid_str, const std::string& common_name) {
//     // crypto_adaptor.Init();????
//     if (http_client_) {
//         return http_client_->CreateClientCsrKeyPair(priv_key_slot_uuid_str, common_name);
//     } else {
//         HTTPS_ERROR << "http_client_ is null";
//         return DataBuffer();
//     }
// }

// DataBuffer PkiRequest::CreateClientCsrKeyPairWithDn(const std::string& priv_key_slot_uuid_str, const DnInfo& dn_info) {
//     hozon::netaos::crypto::CryptoAdapter crypto_adaptor;
//     // crypto_adaptor.Init();?????
//     hozon::netaos::crypto::DnInfo adapter_dn_info;
//     adapter_dn_info.country = dn_info.country;
//     adapter_dn_info.organization = dn_info.organization;
//     adapter_dn_info.organization_unit = dn_info.organization_unit;
//     adapter_dn_info.state = dn_info.state;
//     adapter_dn_info.common_name = dn_info.common_name;
//     adapter_dn_info.email_address = dn_info.email_address;
//     return crypto_adaptor.CreateClientCsrKeyPairWithDn(priv_key_slot_uuid_str, adapter_dn_info);
// }

// bool PkiRequest::ImportCert(DataBuffer certdata, EncodeFormat ef) {
//     if (http_client_) {
//         return http_client_->ImportCert(certdata, ef);
//     } else {
//         HTTPS_ERROR << "http_client_ is null";
//         return false;
//     }
// }

// int PkiRequest::VerifyCert(DataBuffer certdata, EncodeFormat ef) {
//     if (http_client_) {
//         return http_client_->VerifyCert(certdata, ef);
//     } else {
//         HTTPS_ERROR << "http_client_ is null";
//         return -1;
//     }
// }

}  // namespace https
}  // namespace netaos
}  // namespace hozon
