#include "tsp_pki_config.h"
#include <sys/stat.h>
// #include <sys_ctr.h> 
#include <unistd.h>
#include <yaml-cpp/yaml.h>
// #include "hz_fm_agent.h"
#include "tsp_pki_utils.h"
#include "tsp_pki_log.h"


namespace hozon {
namespace netaos {
namespace tsp_pki {

#define CHECK_CONFIG_KEY(config, key) \
    if (!(config[key].IsDefined())) { \
        PKI_ERROR << "Cannot find " << key << " in config file"; \
        return false; \
    }

#define CHECK_RETURN(statement, description) \
    if (!(statement)) { \
        PKI_ERROR << "Config error: " << description; \
    }

const uint32_t CERT_UPDATE_THRESHOLD_DEFAULT = 90;

TspPkiConfig* TspPkiConfig::instance_ = nullptr;
static std::mutex tsp_pki_config_instance_mutex;

TspPkiConfig& TspPkiConfig::Instance() {
    std::lock_guard<std::mutex> lock(tsp_pki_config_instance_mutex);
    if (!instance_) {
        instance_ = new TspPkiConfig;
    }

    return *instance_;
}

void TspPkiConfig::Destroy() {
    std::lock_guard<std::mutex> lock(tsp_pki_config_instance_mutex);
    delete instance_;
    instance_ = nullptr;
}

TspPkiConfig::TspPkiConfig()
: stopped_(false)
, debug_(false)
, cert_update_threshold_(CERT_UPDATE_THRESHOLD_DEFAULT)
, config_file_(""){

    for (int i = 0; i < kDomainNum; ++i) {
        domains_[i] = "";
    }

    for (int i = 0; i < kUrlPathNum; ++i) {
        urls_[i] = "";
    }
}

TspPkiConfig::~TspPkiConfig() {

}

void TspPkiConfig::Start(const std::string& yaml_file) {
    SetConfigYaml(yaml_file);
    ReadConfig();
}

void TspPkiConfig::Stop() {
    stopped_ = true;
}

bool TspPkiConfig::IsDebugOn() {
    return debug_;
}

std::string TspPkiConfig::GetDomain(DomainType domain_type) {
    std::lock_guard<std::mutex> lock(config_mutex_);

    std::string domain;
    if (domain_type <DomainType::kDomainNum) {
        domain = domains_[domain_type];
    }

    return domain;
}

std::string TspPkiConfig::GetUrlPath(UrlPathType path_type) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    std::string path;
    if (path_type < UrlPathType::kUrlPathNum) {
        path = urls_[path_type];
    }

    return path;
}

std::string TspPkiConfig::GetCertPath(CertType cert_type) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    std::string path;
    if (cert_type < CertType::kCertNum) {
        path = certs_[cert_type];
    }

    return path;
}

std::string TspPkiConfig::GetKeyPath(KeyType key_type) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    std::string path;
    if (key_type < KeyType::kKeyNum) {
        path = keys_[key_type];
    }

    return path;
}

std::string TspPkiConfig::GetSlot(SlotType slot_type) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    
    std::string slot;
    if (slot_type < SlotType::kSlotNum) {
        slot = slots_[slot_type];
    }

    return slot;
}

std::string TspPkiConfig::GetVin() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    // TODO: Use the formal vin.
    return dummy_vin_;
}

std::string TspPkiConfig::GetSn() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    // TODO: Use the formal sn.
    return dummy_sn_;
}

std::string TspPkiConfig::GetRunConfPath() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return run_conf_path_;
}

std::string TspPkiConfig::GetPresetCertPath() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return preset_cert_path_;
}

std::string TspPkiConfig::GetPresetKeySlot() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return preset_key_slot_;
}

std::string TspPkiConfig::GetKeySlotCfgFile() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return key_slot_cfg_file_;
}

std::string TspPkiConfig::GetPresetKeySlotCfgFile() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return preset_key_slot_cfg_file_;
}

uint32_t TspPkiConfig::GetUpdateThreshold() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return cert_update_threshold_;
}

void TspPkiConfig::SetVinChangeHandler(VinHandler vin_handler) {
    std::lock_guard<std::mutex> lock(vin_handler_mutex_);
    vin_handler_ = vin_handler;
}

void TspPkiConfig::SetConfigYaml(const std::string& yaml_file)
{
    config_file_ = yaml_file;
}

bool TspPkiConfig::ReadConfig() {
    if (0 != access(config_file_.c_str(), F_OK | R_OK)) {
        PKI_ERROR << "Config file is not exit. error. yaml file: " << config_file_;
        return false;
    }

    PKI_INFO << "[TSP_PKI]:Use config file:"<< config_file_;

    YAML::Node config;
    // Read hz_tsp_pki.yaml
    try {
        std::lock_guard<std::mutex> lock(config_mutex_);
        config = YAML::LoadFile(config_file_);

        // Check all neccessary keys if exist.
        CHECK_CONFIG_KEY(config, "com");

        CHECK_CONFIG_KEY(config["com"], "domains");
        CHECK_CONFIG_KEY(config["com"]["domains"], "ecuDomain");
        CHECK_CONFIG_KEY(config["com"]["domains"], "certDomain");

        CHECK_CONFIG_KEY(config["com"], "urlPaths");
        CHECK_CONFIG_KEY(config["com"]["urlPaths"], "remoteConfig");
        CHECK_CONFIG_KEY(config["com"]["urlPaths"], "uploadToken");
        CHECK_CONFIG_KEY(config["com"]["urlPaths"], "uuid");
        CHECK_CONFIG_KEY(config["com"]["urlPaths"], "applyCert");

        CHECK_CONFIG_KEY(config["com"], "certPaths");
        CHECK_CONFIG_KEY(config["com"]["certPaths"], "deviceCertPath");
        CHECK_CONFIG_KEY(config["com"]["certPaths"], "preintallPfx");

        CHECK_CONFIG_KEY(config["com"], "slots");
        CHECK_CONFIG_KEY(config["com"]["slots"], "slotA");
        CHECK_CONFIG_KEY(config["com"]["slots"], "slotB");

        CHECK_CONFIG_KEY(config["com"], "certUpdateThreshold");

        CHECK_CONFIG_KEY(config, "runConfPath");

        // Read domains config.
        domains_[kDomainTspPkiCert] = config["com"]["domains"]["certDomain"].as<std::string>();
        domains_[kDomainTspPkiEcu] = config["com"]["domains"]["ecuDomain"].as<std::string>();

        // Read urls config.
        urls_[kUrlPathRemoteConfig] = config["com"]["urlPaths"]["remoteConfig"].as<std::string>();
        urls_[kUrlPathUploadToken] = config["com"]["urlPaths"]["uploadToken"].as<std::string>();
        urls_[kUrlPathUuid] = config["com"]["urlPaths"]["uuid"].as<std::string>();
        urls_[kUrlPathApplyCert] = config["com"]["urlPaths"]["applyCert"].as<std::string>();

        // Read cert paths config.
        certs_[kCertDevice] = config["com"]["certPaths"]["deviceCertPath"].as<std::string>();
        certs_[kCertRootCa] = config["com"]["certPaths"]["rootCertPath"].as<std::string>();

        if (config["com"]["certPaths"]["preinstallClientCertPath"].IsDefined()) {
            certs_[kCertPreintallClientCert] = config["com"]["certPaths"]["preinstallClientCertPath"].as<std::string>();
        }

        certs_[kCertPreInstallPfx] = config["com"]["certPaths"]["preintallPfx"].as<std::string>();

        // Read key paths config.
        if (config["com"]["keyPaths"].IsDefined() && config["com"]["keyPaths"]["preinstallKeyPath"].IsDefined()) {
            keys_[kKeyPreinstall] = config["com"]["keyPaths"]["preinstallKeyPath"].as<std::string>();
        }

        // Read slots config.
        slots_[kSlotA] = config["com"]["slots"]["slotA"].as<std::string>();
        slots_[kSlotB] = config["com"]["slots"]["slotB"].as<std::string>();

        // Read certificate update threshold.
        cert_update_threshold_ = config["com"]["certUpdateThreshold"].as<uint32_t>();

        // Read runtime config file path.
        if (config["runConfPath"].IsDefined()) {
            run_conf_path_ = config["runConfPath"].as<std::string>();
        }

        // Read preset pem path.
        if (config["presetCertPath"].IsDefined()) {
            preset_cert_path_ = config["presetCertPath"].as<std::string>();
        }else{
            PKI_WARN << "yaml config file do not has presetCertPath.";
        }

        if (config["presetKeySlot"].IsDefined()) {
            preset_key_slot_ = config["presetKeySlot"].as<std::string>();
        }else{
            PKI_WARN << "yaml config file do not has presetKeySlot.";
        }

        if (config["key_slot_cfg_file"].IsDefined()) {
            key_slot_cfg_file_ = config["key_slot_cfg_file"].as<std::string>();
        }else{
            PKI_WARN << "yaml config file do not has key_slot_cfg_file.";
        }

        if (config["preset_key_slot_cfg_file"].IsDefined()) {
            preset_key_slot_cfg_file_ = config["preset_key_slot_cfg_file"].as<std::string>();
        }else{
            PKI_WARN << "yaml config file do not has preset_key_slot_cfg_file.";
        }

        // Read debug config.
        if (config["debug"].IsDefined() && config["debug"]["enable"].IsDefined()) {
            debug_ = config["debug"]["enable"].as<bool>();
            if (debug_) {
                if (config["debug"]["dummyVin"].IsDefined()) {
                    dummy_vin_ = config["debug"]["dummyVin"].as<std::string>();
                }
                if (config["debug"]["dummyVin"].IsDefined()) {
                    dummy_sn_ = config["debug"]["dummySn"].as<std::string>();
                }
                if (config["debug"]["tsyncbyManual"].IsDefined()) {
                    tsyncbyManual_ = config["debug"]["tsyncbyManual"].as<bool>();
                }
            }
        }

        PKI_INFO << "Parse config DomainTspPkiCert:"<<domains_[kDomainTspPkiCert];
        PKI_INFO << "Parse config successfully.";
    } catch (YAML::ParserException& e) {
        PKI_ERROR << "Exception when parsing hz_tsp_pki.yaml file.Error info:"<<e.what();
        return false;
    }

    return true;
}

}
}
}