/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: pki_request.h is designed for https.
 */
#ifndef PKI_REQUEST_H_
#define PKI_REQUEST_H_
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

// #include "http_client.h"
#include "entry_point.h"
#include "https_types.h"
namespace hozon {
namespace netaos {
namespace https {

class HttpClient;

class PkiRequest {
public:
    PkiRequest(
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
    std::string root_ca_cert_file);

    ~PkiRequest();

    void Init();

    // void Init(std::string client_cert, std::string client_priv_slot);

    /// @brief download single file
    /// @param url URL
    /// @param save_path save download file to save_path
    /// @param  download_callback download callback， return download status and
    /// rate of progress, can be null
    /// @return interface call success or fail
    int Download(ResponseHandler handler);

    /// @brief cancel download process
    /// @return interface call success or fail
    bool CancelDownLoad();

private:
    std::string url_;
    std::string save_file_path_;
    std::map<std::string, std::string> headers_;
    std::string post_data_;
    std::shared_ptr<std::vector<uint8_t>> post_data_v2_;

    int sdkType_;
    std::string client_ap_priv_key_slot_;
    std::string client_priv_key_file_;
    std::string client_cert_chain_;
    std::string client_key_cert_p12_;
    std::string client_key_cert_p12_pass_;
    std::string root_ca_cert_file_;

private:
    std::shared_ptr<HttpClient> http_client_;
    RequestPtr req_ptr;
  
};
}  // namespace https
}  // namespace netaos
}  // namespace hozon
#endif