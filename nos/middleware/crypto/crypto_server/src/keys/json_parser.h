#ifndef ARA_CRYPTO_KEYS_IMP_JSON_PARSER_H_
#define ARA_CRYPTO_KEYS_IMP_JSON_PARSER_H_

#include <iostream>
#include <fstream>
#include <json/json.h>

#include "keys/key_slot_content_props.h"
#include "keys/key_slot_prototype_props.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

class JsonParser {
public:
    static bool parseJson(std::string jsonfile,std::string& uuid,KeySlotPrototypeProps& protoProps,KeySlotContentProps& contentProps);
    static crypto::Uuid convertStringtoUuid(std::string uuidStr);
    static crypto::CryptoAlgId convertStringtoAlgId(std::string str);
    static crypto::CryptoObjectType convertStringtoCryptoObjectType(std::string str);

};

}  // namespace keys
}  // namespace crypto
}  // namespace hozon
}  // namespace neta
#endif  // #define ARA_CRYPTO_KEYS_IMP_JSON_PARSER_H_