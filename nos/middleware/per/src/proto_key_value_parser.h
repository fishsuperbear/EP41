/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: per vs序列化 反序列化
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_SRC_PROTO_KEY_VALUE_PARSER_H_
#define MIDDLEWARE_PER_SRC_PROTO_KEY_VALUE_PARSER_H_

#include <string>

#include "src/per_inner_type.h"

#include "proto/per_key_value.pb.h"

namespace hozon {
namespace netaos {
namespace per {

// 定义一个动态类型消息的解析器
class ProtoKeyValueParser {
 public:
    // 解析一个字节数组为 DynamicMessage 对象

    static bool ParseProtoText(const std::string& data, InnerKeyValueMap& key_value_map);

    // 将 DynamicMessage 对象序列化为字节数组
    static bool SerializeToProtoText(const InnerKeyValueMap& kv_map, std::string& text);
};

}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_SRC_PROTO_KEY_VALUE_PARSER_H_
