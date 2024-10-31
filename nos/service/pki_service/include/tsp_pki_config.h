/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: TspPkiConfig class definition.
 */

#ifndef V2C_TSP_PKI_TSP_PKI_CONFIG_LOCAL_CONFIG_H
#define V2C_TSP_PKI_TSP_PKI_CONFIG_LOCAL_CONFIG_H

#include <string>
#include <mutex>
#include <functional>

namespace hozon {
namespace netaos {
namespace tsp_pki {

enum DomainType {
    kDomainTspPkiCert = 0,
    kDomainTspPkiEcu,
    kDomainNum
};

enum UrlPathType {
    kUrlPathRemoteConfig = 0,
    kUrlPathUploadToken,
    kUrlPathUuid,
    kUrlPathApplyCert,
    kUrlPathNum
};

enum PrivateSlotType {
    kPrivateKeySlotA = 0,
    kPrivateKeySlotB,
    kPrivateKeySlotNum,
};

enum CertType {
    kCertDevice = 0,
    kCertRootCa,
    kCertPreintallClientCert,
    kCertPreInstallPfx,
    kCertNum
};

enum KeyType {
    kKeyPreinstall = 0,
    kKeyNum
};

enum SlotType {
    kSlotA = 0,
    kSlotB,
    kSlotNum
};

using VinHandler = std::function<void()>;
class TspPkiConfig {
public:

    static TspPkiConfig& Instance();
    static void Destroy();

    void Start(const std::string& yaml_file);
    void Stop();
    bool IsDebugOn();
    std::string GetDomain(DomainType domain_type);
    std::string GetUrlPath(UrlPathType path_type);
    std::string GetCertPath(CertType cert_type);
    std::string GetKeyPath(KeyType key_type);
    std::string GetSlot(SlotType slot_type);
    std::string GetVin();
    std::string GetSn();
    std::string GetRunConfPath();
    std::string GetPresetCertPath();
    std::string GetPresetKeySlot();
    std::string GetKeySlotCfgFile();
    std::string GetPresetKeySlotCfgFile();
    uint32_t GetUpdateThreshold();
    void SetVinChangeHandler(VinHandler vin_handler);
    bool IsTsyncbyManual(){
        return tsyncbyManual_;
    };
    void SetConfigYaml(const std::string& yaml_file);
    bool ReadConfig();

private:
    TspPkiConfig();
    TspPkiConfig(const TspPkiConfig& cfg){};
    ~TspPkiConfig();

    bool stopped_;
    bool debug_;
    std::string config_file_;

    std::string domains_[kDomainNum];
    std::string urls_[kUrlPathNum];
    std::string certs_[kCertNum];
    std::string keys_[kKeyNum];
    std::string slots_[kPrivateKeySlotNum];
    std::string dummy_vin_;
    std::string dummy_sn_;
    std::string run_conf_path_;
    std::string preset_cert_path_;
    std::string preset_key_slot_;
    std::string key_slot_cfg_file_;
    std::string preset_key_slot_cfg_file_;
    uint32_t cert_update_threshold_;
    bool tsyncbyManual_ = false;

    VinHandler vin_handler_;

    std::mutex config_mutex_;
    std::mutex vin_handler_mutex_;

    static TspPkiConfig* instance_;
};

}
}
}

#endif