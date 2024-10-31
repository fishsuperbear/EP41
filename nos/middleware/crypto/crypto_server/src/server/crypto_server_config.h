/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: TspPkiConfig class definition.
 */

#ifndef CRYPTO_SERVER_CONFIG_H
#define CRYPTO_SERVER_CONFIG_H

#include <string>
#include <mutex>

namespace hozon {
namespace netaos {
namespace crypto {

class CryptoConfig {
   public:
    enum CertType { kCertDevice = 0, kCertJitDevice, kCertRootCa, kCertPreintallClientCert, kCertPreInstallPfx, kCertNum };

    enum KeyType { kKeyPreinstall = 0, kKeyNum };

    enum SlotType { kSlotA = 0, kSlotB, kSlotNum };

    static CryptoConfig& Instance();
    std::string GetOemPresetKeyFile(){
        return oem_preset_key_file_;
    };
    std::string GetOemPresetKeyFile_En(){
        return oem_preset_key_file_en_;
    };
    std::string GetKeySlotCfgFile(){
        return key_slot_cfg_file_;
    };
    std::string GetKeysStoragePath(){
        return keys_storage_path_;
    };
    std::string GetDeviceCertPath(){
        return device_cert_path_;
    };

   private:
    CryptoConfig();
    CryptoConfig(const CryptoConfig& cfg){};
    ~CryptoConfig();
    bool ReadConfig();

    bool stopped_ = false;
    std::string config_file_;

    std::string oem_preset_key_file_;
    std::string oem_preset_key_file_en_;
    std::string key_slot_cfg_file_;
    std::string keys_storage_path_;
    std::string device_cert_path_;

    std::mutex config_mutex_;
    static CryptoConfig* instance_;
};

}
}
}

#endif