/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: global enum value header
 * Create: 2019-6-25
 */

#ifndef ARA_GODEL_COMMON_JSONPARSER_GLOBAL_H
#define ARA_GODEL_COMMON_JSONPARSER_GLOBAL_H
#include <cstdint>

namespace ara        {
namespace godel      {
namespace common     {
namespace jsonParser {
enum class JsonType: uint8_t {
    JSON_UNDEFINED = 0U,
    JSON_NULL,
    JSON_FALSE,
    JSON_TRUE,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT,
    JSON_DEFAULT
};

enum class JsonParseValue: uint8_t {
    JSON_PARSE_UNDEFINED = 0U,
    JSON_PARSE_OK,
    JSON_PARSE_NUMBER_TOO_BIG,
    JSON_PARSE_EXPECT_VALUE,
    JSON_PARSE_INVALID_VALUE,
    JSON_PARSE_ROOT_NOT_SINGULAR,
    JSON_PARSE_MISS_KEY,
    JSON_PARSE_MISS_COLON,
    JSON_PARSE_MISS_COMMA_OR_CURLY_BRACKET,
    JSON_PARSE_MISS_COMMA_OR_SQUARE_BRACKET,
    JSON_PARSE_MISS_QUOTATION_MARK,
    JSON_PARSE_CRC_VERIFY_FAIL
};

enum class CRCVerificationType: uint8_t {
    NOT = 0U,
    WEAK,
    STRONG
};
} // namespace jsonParser
} // namespace common
} // namespace godel
} // namespace ara
#endif // ARA_GODEL_COMMON_JSONPARSER_GLOBAL_H_
