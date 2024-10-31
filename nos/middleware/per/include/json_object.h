
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: json序列化，反序列化
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_PER_INCLUDE_JSON_OBJECT_H_
#define MIDDLEWARE_PER_INCLUDE_JSON_OBJECT_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include <json/json.h>

#include "core/result.h"
#include "core/span.h"
#include "kvs_type.h"
#include "per_error_domain.h"
#include "struct2x/json/decoder.h"  // decode（json -> struct）
#include "struct2x/json/encoder.h"  // encode（struct -> json）
#include "struct2x/struct2x.h"      // SERIALIZE

namespace hozon {
namespace netaos {
namespace per {

class JsonObject {
 public:
    JsonObject() {}
    virtual ~JsonObject() {}
    static JsonObject& GetInstance() {
        static JsonObject instance;
        return instance;
    }
    JsonObject(const JsonObject&);
    JsonObject& operator=(const JsonObject&);
    template <typename T>
    hozon::netaos::core::Result<std::string> SerializeObject(const T& data, bool tostyple) noexcept {
        std::string strJson;
        bool bEncode = struct2x::JSONEncoder(strJson) << data;
        PER_LOG_INFO << "SerializeObject:" << strJson.size() << " " << bEncode;
        if (!bEncode) {
            return hozon::netaos::core::Result<std::string>::FromError(PerErrc::kSerializeError);
        }
        if (tostyple) {
            Json::CharReaderBuilder builder;
            Json::CharReader* reader(builder.newCharReader());
            Json::Value root;
            JSONCPP_STRING errs;
            if (reader->parse(strJson.c_str(), strJson.c_str() + strJson.length(), &root, &errs)) {
                strJson = root.toStyledString();
            } else {
                PER_LOG_ERROR << "Json buffer parse error: " << errs;
                return hozon::netaos::core::Result<std::string>::FromError(PerErrc::kIntegrityError);
            }
            if (reader != nullptr) {
                delete reader;
                reader = nullptr;
            } else {
                PER_LOG_ERROR << "reader is null  ";
            }
        }
        return hozon::netaos::core::Result<std::string>::FromValue(strJson);
    }
    template <typename T>
    hozon::netaos::core::Result<T> DerializeObject(const std::string& data) noexcept {
        T insDecode;
        bool bDecode = struct2x::JSONDecoder(data.c_str(), data.size()) >> insDecode;
        PER_LOG_INFO << "DerializeObject:" << data.size() << " " << bDecode;
        if (!bDecode) {
            return hozon::netaos::core::Result<T>::FromError(PerErrc::kDeSerializeError);
        }
        return hozon::netaos::core::Result<T>::FromValue(insDecode);
    }
};  // namespace per
}  // namespace per
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_PER_INCLUDE_JSON_OBJECT_H_
