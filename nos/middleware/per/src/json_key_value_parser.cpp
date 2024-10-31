/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: per vs序列化 反序列化
 * Created on: Feb 7, 2023
 *
 */
#include "src/json_key_value_parser.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "include/kvs_type.h"
#include "src/per_inner_type.h"

namespace hozon {
namespace netaos {
namespace per {

// 解析一个字节数组为 DynamicMessage 对象
bool JsonKeyValueParser::ParseJsonText(const std::string& data, InnerKeyValueVec& key_vec) {
    bool res = false;
    hozon::netaos::core::Result<InnerKeyValueVec> result = JsonObject::GetInstance().DerializeObject<InnerKeyValueVec>(data);
    if (result.HasValue()) {
        key_vec = result.Value();
        res = true;
    }
    PER_LOG_INFO << "res :" << res;
    return res;
}

// 将 DynamicMessage 对象序列化为字节数组
bool JsonKeyValueParser::SerializeToJsonText(const InnerKeyValueVec& kv_vec, std::string& text) {
    bool res = false;
    hozon::netaos::core::Result<std::string> result = JsonObject::GetInstance().SerializeObject(kv_vec, true);
    if (result.HasValue()) {
        text = result.Value();
        res = true;
    }
    PER_LOG_INFO << "res :" << res;
    return res;
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
