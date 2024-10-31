/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: JsonKeyValueStorage class header
 *              This class provides json key value storage related functions
 * Create: 2019-06-25
 */

#ifndef ARA_GODEL_COMMON_KVS_JSON_KEY_VALUE_STORAGE_H
#define ARA_GODEL_COMMON_KVS_JSON_KEY_VALUE_STORAGE_H

#include "key_value_storage.h"

namespace ara    {
namespace godel  {
namespace common {
namespace kvs    {
enum class KvsLoadValue: uint8_t {
    OK,
    CRC_VERIFY_ERROR,
    JSON_PARSE_ERROR,
    WRONG_KVS_FORMAT
};
enum class CRCVerificationType: uint8_t {
    NOT = 0U,
    WEAK,
    STRONG
};
class JsonKeyValueStorage {
public:
    static bool LoadFromJson(std::string const &path, KeyValueStorage& kvsObj);
    static KvsLoadValue LoadFromJson(std::string const &path, KeyValueStorage& kvsObj, CRCVerificationType const type);

private:
};
} // namespace kvs
} // namespace common
} // namespace godel
} // namespace ara
#endif // ARA_GODEL_COMMON_KVS_JSON_KEY_VALUE_STORAGE_H_ */
