#include <openssl/pem.h>
#include <iostream>
#include <filesystem>
#include "import_key.h"
#include "cryp/cryobj/imp_private_key.h"
#include "cryp/cryobj/simpl_symmetric_key.h"
#include "encryption_service.hpp"
#include "common/crypto_logger.hpp"
#include "common/base_id_types.h"
#include "utility/crypto_adapter.h"
#include "crypto_server_config.h"
#include "keys/json_parser.h"
#include "crypto_server_config.h"

namespace hozon {
namespace netaos {
namespace crypto {

ImportKey::ImportKey() {

}

int ImportKey::DecryptYaml(std::string sourch, std::string destion) {
    if(!EncryptionService::Instance().do_FileCrypt(sourch, destion, 0)) {
        return 0;
    }
    return 1;
}

int ImportKey::SaveKeyFromYaml(const std::string yamlPath) {
    // 1.parse yaml
    std::map<std::string, ImportKeyInfo> keyString_Map;
    if (!ParseYaml(yamlPath, keyString_Map)) {
        return 0;
    }

    // 2. construct private key obj & save
    int res = constructAndSaveKey(keyString_Map);
    return res;
}

int ImportKey::ParseYaml(std::string yamlPath, std::map<std::string, ImportKeyInfo>& keyMap) {
    YAML::Node config;
    std::map<std::string, ImportKeyInfo> keyString_Map;
    if (!std::filesystem::exists(yamlPath)) {
       CRYP_ERROR << yamlPath << " does not exist!";
        return 0;
    }
    std::string decryptedYamlData = "";
    if(!EncryptionService::Instance().do_FileCrypt_ReturnString(yamlPath, decryptedYamlData, 0)) {
        return 0;
    }
    if (decryptedYamlData.empty()) {
        return 0;
    }
    try {
        config =YAML::Load(decryptedYamlData);

        auto keys = config["keys"];
        CRYP_INFO << keys.size() << " keys in config.";
        for (const auto& key : keys) {
            std::string name = key["name"].as<std::string>();
            if(key["name"].IsDefined()
                && key["alg_id"].IsDefined()
                && key["value"].IsDefined()
                && key["slot_uuid"].IsDefined()
                && key["hash256"].IsDefined()
                && key["objectUid"].IsDefined()
                && key["objectType"].IsDefined()
                && key["isSession"].IsDefined()
                && key["isExportable"].IsDefined()
                && key["objectSize"].IsDefined()
                && key["allowedUsageFlags"].IsDefined()) {

            } else {
                CRYP_ERROR << "load YAML, but some property is lost, please check !";
                return 0;
            }

            keyString_Map[name].name = key["name"].as<std::string>();
            keyString_Map[name].alg_id = key["alg_id"].as<std::string>();
            keyString_Map[name].value = key["value"].as<std::string>();
            keyString_Map[name].slot_uuid = key["slot_uuid"].as<std::string>();
            keyString_Map[name].hash256 = key["hash256"].as<std::string>();

            std::string str_obj_uid = key["objectUid"].as<std::string>();
            auto obj_uid = keys::JsonParser::convertStringtoUuid(str_obj_uid);
            keyString_Map[name].object_info.objectUid.mCouid.mGeneratorUid.mQwordMs = obj_uid.mQwordMs;
            keyString_Map[name].object_info.objectUid.mCouid.mGeneratorUid.mQwordLs = obj_uid.mQwordLs;

            keyString_Map[name].object_info.objectUid.mCOType = keys::JsonParser::convertStringtoCryptoObjectType( key["objectType"].as<std::string>());
            keyString_Map[name].object_info.isSession = key["isSession"].as<bool>();
            keyString_Map[name].object_info.isExportable = key["isExportable"].as<bool>();
            keyString_Map[name].object_info.payloadSize = key["objectSize"].as<int>();
            keyString_Map[name].allowed_usage = key["allowedUsageFlags"].as<int>();

            // 输出获取到的数据
            CRYP_INFO << "Parse config of imported keys. \nName: "
                      << keyString_Map[name].name
                      << ", Alg_id: " << keyString_Map[name].alg_id
                      << ", ObjectSize: " << keyString_Map[name].object_info.payloadSize
                      << ", ObjectType: " << static_cast<int>(keyString_Map[name].object_info.objectUid.mCOType)
                      << ", Value: " << keyString_Map[name].value
                      << ", Slot UUID: " << keyString_Map[name].slot_uuid
                      << ", Hash256: " << keyString_Map[name].hash256
                      << ", ObjectUid: " << keyString_Map[name].object_info.objectUid.mCouid.mGeneratorUid.ToUuidStr()
                      << ", Allowed_usage: " << keyString_Map[name].allowed_usage;
        }
    } catch (YAML::ParserException& e) {
        CRYP_ERROR << "yaml error:"<< e.what();
        return 0;
    }
    keyMap = keyString_Map;
    return 1;
}

int ImportKey::constructAndSaveKey(std::map<std::string, ImportKeyInfo>& map_KeyStr) {
    CryptoAdapter adpt;
    std::string key_slot_cfg_file = CryptoConfig::Instance().GetKeySlotCfgFile();
    adpt.Init(key_slot_cfg_file);
    for (auto keyString : map_KeyStr) {
        std::string alg_id = keyString.second.alg_id;
        CRYP_INFO << "key name : "<< keyString.first << " uuid :" << keyString.second.slot_uuid << " alg_id:" <<alg_id;
        cryp::CryptoPrimitiveId primitive_id(keys::JsonParser::convertStringtoAlgId(alg_id));
        if (0 == keyString.first.compare("rsa_private_key_uat") || 0 == keyString.first.compare("rsa_private_key_pro")) {
            // 创建内存BIO对象并将PEM格式的私钥字符串写入
            BIO* privateKeyBio = BIO_new_mem_buf(keyString.second.value.c_str(), keyString.second.value.length());

            // 从内存BIO中读取并构造私钥对象
            EVP_PKEY* pkey = PEM_read_bio_PrivateKey(privateKeyBio, NULL, NULL, NULL);

            // 检查私钥对象是否创建成功
            if (pkey == NULL) {
                ERR_print_errors_fp(stdout);
                CRYP_ERROR << "Failed to read private key.";
                BIO_free(privateKeyBio);
                return 0;
            }

            cryp::ImpPrivateKey::Uptrc uptrc = std::make_unique<const cryp::ImpPrivateKey>(pkey, keyString.second.object_info, primitive_id, keyString.second.allowed_usage);
            adpt.SavePrivateKey(*dynamic_cast<cryp::PrivateKey*>(const_cast<cryp::ImpPrivateKey*>(uptrc.release())), keyString.second.slot_uuid);
            // 释放资源
            EVP_PKEY_free(pkey);
            BIO_free(privateKeyBio);
        }
        if (0 == keyString.first.compare("common_aes_key") || 0 == keyString.first.compare("27_app_mask") || 0 == keyString.first.compare("27_boot_mask")) {
            std::vector<uint8_t> pkey;
            // 将string的内容转换为uint8_t并写入到vector中
            pkey.insert(pkey.end(), keyString.second.value.begin(), keyString.second.value.end());
            cryp::SimplSymmetricKey::Uptrc uptrc = std::make_unique<const cryp::SimplSymmetricKey>(pkey, keyString.second.object_info, primitive_id, keyString.second.allowed_usage);
            adpt.SaveSymmetricKey(*dynamic_cast<cryp::SymmetricKey*>(const_cast<cryp::SymmetricKey*>(uptrc.release())), keyString.second.slot_uuid);
        }
    }
    return 1;
}

}
}
}