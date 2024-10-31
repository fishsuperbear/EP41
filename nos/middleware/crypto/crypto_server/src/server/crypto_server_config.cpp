#include "crypto_server_config.h"
#include <sys/stat.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include "common/crypto_logger.hpp"



namespace hozon {
namespace netaos {
namespace crypto {

#define CHECK_CONFIG_KEY(config, key) \
    if (!(config[key].IsDefined())) { \
        CRYP_ERROR << "Cannot find " << key << " in config file"; \
        return false; \
    }


CryptoConfig* CryptoConfig::instance_ = nullptr;
static std::mutex crypto_server_config_instance_mutex;

CryptoConfig& CryptoConfig::Instance() {
    std::lock_guard<std::mutex> lock(crypto_server_config_instance_mutex);
    if (!instance_) {
        instance_ = new CryptoConfig;
    }
    return *instance_;
}

CryptoConfig::CryptoConfig(){
    // char buf[1024] = {0};
    // if (!getcwd(buf, sizeof(buf))) {
    //     CRYP_INFO << "getcwd failed.";
    //     return;
    // }
    // std::string wk(buf);
    config_file_ = "/app/runtime_service/crypto_server/conf/crypto_server.yaml";
    CRYP_INFO <<"cryptoServer config_file_:" <<config_file_;
    ReadConfig();
}

CryptoConfig::~CryptoConfig() {
    delete instance_;
    instance_ = nullptr;
}

bool CryptoConfig::ReadConfig() {
    std::ifstream yamlfile(config_file_,std::ios::in);
    if(!yamlfile.is_open()){
        CRYP_ERROR << config_file_ << " is not exist.";
        return false;
    }
    CRYP_INFO << "[crypto_server]:Use config file:"<< config_file_;

    YAML::Node config;
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        config = YAML::LoadFile(config_file_);
        CHECK_CONFIG_KEY(config, "storage");

        CHECK_CONFIG_KEY(config["storage"], "keys");
        CHECK_CONFIG_KEY(config["storage"]["keys"], "oem_preset_key_file");
        CHECK_CONFIG_KEY(config["storage"]["keys"], "oem_preset_key_file_en");
        CHECK_CONFIG_KEY(config["storage"]["keys"], "key_slot_cfg_file");
        CHECK_CONFIG_KEY(config["storage"]["keys"], "keys_storage_path");

        CHECK_CONFIG_KEY(config["storage"], "x509");
        CHECK_CONFIG_KEY(config["storage"]["x509"], "device_cert_path");

        // Read config.
        oem_preset_key_file_ = config["storage"]["keys"]["oem_preset_key_file"].as<std::string>();
        oem_preset_key_file_en_ = config["storage"]["keys"]["oem_preset_key_file_en"].as<std::string>();
        key_slot_cfg_file_ = config["storage"]["keys"]["key_slot_cfg_file"].as<std::string>();
        keys_storage_path_ = config["storage"]["keys"]["keys_storage_path"].as<std::string>();
        device_cert_path_ = config["storage"]["x509"]["device_cert_path"].as<std::string>();
        CRYP_INFO << "Parse config successfully.";
    } catch (YAML::ParserException& e) {
        CRYP_ERROR << "Exception when parsing hz_tsp_pki.yaml file.Error info:"<<e.what();
        return false;
    }
    return true;
}

}
}
}