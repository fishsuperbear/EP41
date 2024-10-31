

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: per内部类型
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_SRC_PER_INNER_TYPE_H_
#define MIDDLEWARE_PER_SRC_PER_INNER_TYPE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kvs_type.h"
#include "struct2x/struct2x.h"  // SERIALIZE
namespace hozon {
namespace netaos {
namespace per {

struct InnerValue {
    int value_type;
    std::vector<uint8_t> buf;
    std::vector<int32_t> int32buf;
    std::vector<float> floatbuf;
    std::vector<double> doublebuf;
    std::vector<bool> boolbuf;
    std::vector<std::string> stringbuf;
    std::map<std::string, InnerValue> kv_map;
};

using InnerKeyValueMap = std::map<std::string, InnerValue>;

struct Value {
    JsonType type;
    std::string string;
    Value() {
        string.clear();
        type = JsonType::JSON_PER_TYPE_UNKOWN;
    }
    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, type, string);
    }
};
struct InnerKeyValue {
    std::string key;
    Value value;
    InnerKeyValue() { key.clear(); }
    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, key, value);
    }
};

struct InnerKeyValueVec {
    std::vector<InnerKeyValue> kv_vec;
    InnerKeyValueVec() { kv_vec.clear(); }
    template <typename T>
    void serialize(T& t) {
        SERIALIZE(t, kv_vec);
    }
};

}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_SRC_PER_INNER_TYPE_H_
