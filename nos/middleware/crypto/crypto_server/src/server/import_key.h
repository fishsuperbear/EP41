#pragma once
#include <string>
#include "yaml-cpp/yaml.h"
#include "cryp/cryobj/crypto_object.h"

namespace hozon {
namespace netaos {
namespace crypto {

struct ImportKeyInfo {
    std::string name;
    std::string alg_id;
    int object_type;
    std::string value;
    std::string slot_uuid;
    std::string hash256;
    cryp::CryptoObject::CryptoObjectInfo object_info;
    cryp::CryptoPrimitiveId primitive_id;
    AllowedUsageFlags allowed_usage;
};

class ImportKey {

public:
    ImportKey();
    int DecryptYaml(std::string sourch, std::string destion);
    int SaveKeyFromYaml(std::string yamlPath);

private:
    int ParseYaml(std::string yamlPath, std::map<std::string, ImportKeyInfo>& keyMap);
    int constructAndSaveKey(std::map<std::string, ImportKeyInfo>& map_privateKeyStr);
};

}
}
}