/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-10-18 15:53:16
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-10-19 14:30:49
 * @FilePath: /nos/test/per_test/src/key_value.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef _KEYVALUE_DATA_H__
#define _KEYVALUE_DATA_H__

#include <stdint.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "struct2x/struct2x.h"  // SERIALIZE
enum JsonPerType {
    JSON_PER_TYPE_UNKOWN,
    JSON_PER_TYPE_INT,
    JSON_PER_TYPE_UINT64,
    JSON_PER_TYPE_BOOL,
    JSON_PER_TYPE_FLOAT,
    JSON_PER_TYPE_STRING,
    JSON_PER_TYPE_DOUBLE,
    JSON_PER_TYPE_VEC_INT,
    JSON_PER_TYPE_VEC_BOOL,
    JSON_PER_TYPE_VEC_FLOAT,
    JSON_PER_TYPE_VEC_STRING,
    JSON_PER_TYPE_VEC_DOUBLE
};
struct Value {
    JsonPerType type;
    std::string string;
    Value() {
        string.clear();
        type = JSON_PER_TYPE_UNKOWN;
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

#endif
