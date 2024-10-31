/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: https.h is designed for https.
 */
#ifndef OTA_DOWNLOAD_H_
#define OTA_DOWNLOAD_H_
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <atomic>


// #include "http_client.h"
#include "entry_point.h"
#include "https_types.h"
namespace hozon {
namespace netaos {
namespace https {

// enum class Status : std::uint32_t {
//   init = 0,     // prepare to download file.
//   running = 1,  // running
//   suspend = 2,  // download process suspend
//   fail = 3,     // download process fail
//   complete = 4  // download process complete
// };

class HttpsDownloadInfo {
public:
    // Status status;
    std::uint32_t rate_of_download;
    std::uint32_t reserve1;
    std::uint32_t reserve2;
    std::string reserve3;
    std::string reserve4;
    HttpsDownloadInfo();
};

class HttpClient;

class OtaDownload {
public:

    static OtaDownload& GetInstance() {
        static OtaDownload instance;
        return instance;
    }
    explicit OtaDownload(std::map<std::string, std::string> param_map);

    OtaDownload();

    ~OtaDownload();

    void Destroy();

    void Init();

    // void Init(std::string client_cert, std::string client_priv_slot);

    /// @brief download single file
    /// @param url URL
    /// @param save_path save download file to save_path
    /// @param  download_callback download callback， return download status and
    /// rate of progress, can be null
    /// @return interface call success or fail
    int Download(RequestPtr req_ptr, ResponseHandler handler);

    /// @brief query download info
    /// @param respInfo
    /// @return interface call success or fail
    bool QueryDownloadInfo(std::vector<Response>& respInfo);

    /// @brief stop download process
    /// @return interface call success or fail
    bool StopDownLoad();

    /// @brief restart download process
    /// @return interface call success or fail
    bool ReStartDownLoad();

    /// @brief cancel download process
    /// @return interface call success or fail
    bool CancelDownLoad();

    /// @brief verify download file
    /// @param file_path file path, if null, use save_path combined file name
    /// @return interface call success or fail
    bool Verify(const std::string file_path);

    /// @brief set download service param
    /// @param update_param_map update params
    /// @return
    bool SetParam(const std::map<std::string, std::string> update_param_map);

    /// @brief sign file with sha256 and inner private key
    /// @param file_path input file to be signed
    /// @param sign_file_path output file of the signature
    /// @return
    bool Sign(const std::string file_path, const std::string sign_file_path);

    // bool ImportCert(DataBuffer certdata, EncodeFormat ef);

    // int VerifyCert(DataBuffer certdata, EncodeFormat ef);

    // // bool ImportCrl(DataBuffer crldata);

    // hozon::netaos::crypto::x509::Certificate::Uptrc FindIssuerCert(const hozon::netaos::crypto::x509::X509DN& subject_dn, const hozon::netaos::crypto::x509::X509DN& issuer_dn);
    
    // DataBuffer CreateClientCsrKeyPair(const std::string& priv_key_slot_uuid_str, const std::string& common_name);
    
    // DataBuffer CreateClientCsrKeyPairWithDn(const std::string& priv_key_slot_uuid_str, const DnInfo& dn_info);

private:
  
    std::shared_ptr<HttpClient> http_client_;

    std::map<std::string, std::string> param_map;

    std::once_flag cancelFlag;

    std::string GetVerifyFilePath();

    std::string HexToString(const char* data, int size);

    int StringToHex(const char* bytes, int size, std::unique_ptr<char[]>& uptr,
                    int* outlen);

    bool Unzip(std::string file_path, std::string unzip_path);

    bool VerifySignByJson(std::string file_path, std::string root_path);

    bool VerifyHashByJson(std::string file_path, std::string root_path);

    bool VerifyHash(std::string file_path, std::string hash_value,
                    std::string digest_meth);

    bool VerifySign(std::string file_path, std::string sign_value);

    bool CleanTmpDir(std::string root_path);

    void InitParam();
    std::mutex init_mutex_;
    int32_t req_id_{0};
    static std::atomic_bool initFlag_;

};
}  // namespace https
}  // namespace netaos
}  // namespace hozon
#endif