/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: per vs序列化 反序列化
 * Created on: Feb 7, 2023
 *
 */
#include "src/proto_key_value_parser.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "include/kvs_type.h"
#include "src/per_inner_type.h"

#include "proto/per_key_value.pb.h"

namespace hozon {
namespace netaos {
namespace per {

bool ParseKvs(const proto::KeyValueStorageData& message, InnerKeyValueMap& kv_map) {
    for (auto it = message.fields().begin(); it != message.fields().end(); ++it) {
        PER_LOG_INFO << " value type. " << it->second.value_case();
        switch (it->second.value_case()) {
            case proto::Value::kIntValue: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_INT;
                value.buf.resize(sizeof(int32_t));
                *reinterpret_cast<int32_t*>(value.buf.data()) = it->second.int_value();
                kv_map[it->first] = value;
            } break;
            case proto::Value::kUint64Value: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_UINT64;
                value.buf.resize(sizeof(uint64_t));
                *reinterpret_cast<uint64_t*>(value.buf.data()) = it->second.uint64_value();
                kv_map[it->first] = value;
            } break;
            case proto::Value::kFloatValue: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_FLOAT;
                value.buf.resize(sizeof(float));
                *reinterpret_cast<float*>(value.buf.data()) = it->second.float_value();
                kv_map[it->first] = value;
            } break;
            case proto::Value::kDoubleValue: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_DOUBLE;
                value.buf.resize(sizeof(double));
                *reinterpret_cast<double*>(value.buf.data()) = it->second.double_value();
                kv_map[it->first] = value;
            } break;
            case proto::Value::kBoolValue: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_BOOL;
                value.buf.resize(sizeof(bool));
                *reinterpret_cast<bool*>(value.buf.data()) = it->second.bool_value();
                kv_map[it->first] = value;
            } break;
            case proto::Value::kStringValue: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_STRING;
                value.buf.resize(it->second.string_value().size());
                ::memcpy(value.buf.data(), it->second.string_value().data(), it->second.string_value().size());
                kv_map[it->first] = value;
            } break;
            case proto::Value::kNestedMessage: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_NESTED_MESSAGE;
                value.buf.resize(it->second.nested_message().size());
                ::memcpy(value.buf.data(), it->second.nested_message().data(), it->second.nested_message().size());
                kv_map[it->first] = value;
            } break;
            case proto::Value::kNestedKvs: {
                InnerValue value;
                value.value_type = PROTO_PER_TYPE_NESTED_KVS;
                if (!ParseKvs(it->second.nested_kvs(), value.kv_map)) {
                    PER_LOG_INFO << "parse nested kvs failed.\n";
                    continue;
                }
                kv_map[it->first] = value;
            } break;
            case proto::Value::VALUE_NOT_SET:
            default: {
                if (it->second.vec_bool_val_size()) {
                    InnerValue value;
                    for (int32_t index = 0; index < it->second.vec_bool_val_size(); index++) {
                        value.value_type = PROTO_PER_TYPE_VEC_BOOL;
                        value.boolbuf.push_back(it->second.vec_bool_val(index));
                    }
                    kv_map[it->first] = value;
                }
                if (it->second.vec_double_val_size()) {
                    InnerValue value;
                    for (int32_t index = 0; index < it->second.vec_double_val_size(); index++) {
                        value.value_type = PROTO_PER_TYPE_VEC_DOUBLE;
                        value.doublebuf.push_back(it->second.vec_double_val(index));
                    }
                    kv_map[it->first] = value;
                }
                if (it->second.vec_float_val_size()) {
                    InnerValue value;
                    for (int32_t index = 0; index < it->second.vec_float_val_size(); index++) {
                        value.value_type = PROTO_PER_TYPE_VEC_FLOAT;
                        value.floatbuf.push_back(it->second.vec_float_val(index));
                    }
                    kv_map[it->first] = value;
                }
                if (it->second.vec_string_val_size()) {
                    InnerValue value;
                    for (int32_t index = 0; index < it->second.vec_string_val_size(); index++) {
                        value.value_type = PROTO_PER_TYPE_VEC_STRING;
                        value.stringbuf.push_back(it->second.vec_string_val(index));
                    }
                    kv_map[it->first] = value;
                }
                if (it->second.vec_int_val_size()) {
                    InnerValue value;
                    for (int32_t index = 0; index < it->second.vec_int_val_size(); index++) {
                        value.value_type = PROTO_PER_TYPE_VEC_INT;
                        value.int32buf.push_back(it->second.vec_int_val(index));
                    }
                    kv_map[it->first] = value;
                }
            } break;
        }
    }

    return true;
}

// 解析一个字节数组为 DynamicMessage 对象
bool ProtoKeyValueParser::ParseProtoText(const std::string& data, InnerKeyValueMap& key_value_map) {
    bool res = false;
    proto::KeyValueStorageData message;
    if (google::protobuf::TextFormat::ParseFromString(data, &message)) {
        res = ParseKvs(message, key_value_map);
    }
    PER_LOG_INFO << "res :" << res;
    return res;
}

bool SerializeKvs(const InnerKeyValueMap& kv_map, proto::KeyValueStorageData& message) {
    for (auto it = kv_map.begin(); it != kv_map.end(); ++it) {
        PER_LOG_INFO << " value type. " << it->second.value_type;
        // proto::Value value;
        switch (it->second.value_type) {
            case PROTO_PER_TYPE_INT: {
                proto::Value value;
                value.set_int_value(*reinterpret_cast<const int32_t*>(it->second.buf.data()));
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_UINT64: {
                proto::Value value;
                value.set_uint64_value(*reinterpret_cast<const uint64_t*>(it->second.buf.data()));
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_FLOAT: {
                proto::Value value;
                value.set_float_value(*reinterpret_cast<const float*>(it->second.buf.data()));
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_DOUBLE: {
                proto::Value value;
                value.set_double_value(*reinterpret_cast<const double*>(it->second.buf.data()));
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_BOOL: {
                proto::Value value;
                value.set_bool_value(*reinterpret_cast<const bool*>(it->second.buf.data()));
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_STRING: {
                proto::Value value;
                value.set_string_value(std::string(reinterpret_cast<const char*>(it->second.buf.data()), it->second.buf.size()));
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_NESTED_MESSAGE: {
                proto::Value value;
                value.set_nested_message(std::string(reinterpret_cast<const char*>(it->second.buf.data()), it->second.buf.size()));
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_NESTED_KVS: {
                proto::Value value;
                auto nested_kvs = value.mutable_nested_kvs();
                if (!SerializeKvs(it->second.kv_map, *nested_kvs)) {
                    PER_LOG_INFO << "Convert key value to proto message failed.\n";
                    return false;
                }
                (*(message.mutable_fields()))[it->first] = value;
            } break;
            case PROTO_PER_TYPE_VEC_INT: {
                if (it->second.int32buf.size()) {
                    proto::Value value;
                    value.clear_vec_int_val();
                    for (size_t index = 0; index < it->second.int32buf.size(); index++) {
                        value.add_vec_int_val(it->second.int32buf[index]);
                    }
                    (*(message.mutable_fields()))[it->first] = value;
                }
            } break;
            case PROTO_PER_TYPE_VEC_DOUBLE: {
                if (it->second.doublebuf.size()) {
                    proto::Value value;
                    value.clear_vec_double_val();
                    for (size_t index = 0; index < it->second.doublebuf.size(); index++) {
                        value.add_vec_double_val(it->second.doublebuf[index]);
                    }
                    (*(message.mutable_fields()))[it->first] = value;
                }
            } break;
            case PROTO_PER_TYPE_VEC_BOOL: {
                if (it->second.boolbuf.size()) {
                    proto::Value value;
                    for (size_t index = 0; index < it->second.boolbuf.size(); index++) {
                        value.add_vec_bool_val(it->second.boolbuf[index]);
                    }
                    (*(message.mutable_fields()))[it->first] = value;
                }
            } break;
            case PROTO_PER_TYPE_VEC_FLOAT: {
                if (it->second.floatbuf.size()) {
                    proto::Value value;
                    value.clear_vec_float_val();
                    for (size_t index = 0; index < it->second.floatbuf.size(); index++) {
                        value.add_vec_float_val(it->second.floatbuf[index]);
                    }
                    (*(message.mutable_fields()))[it->first] = value;
                }
            } break;
            case PROTO_PER_TYPE_VEC_STRING: {
                if (it->second.stringbuf.size()) {
                    proto::Value value;
                    value.clear_vec_string_val();
                    for (size_t index = 0; index < it->second.stringbuf.size(); index++) {
                        value.add_vec_string_val(it->second.stringbuf[index]);
                    }
                    (*(message.mutable_fields()))[it->first] = value;
                }
            } break;
            default:
                PER_LOG_INFO << "unkown value type.\n";
                break;
        }
    }

    return true;
}

// 将 DynamicMessage 对象序列化为字节数组
bool ProtoKeyValueParser::SerializeToProtoText(const InnerKeyValueMap& kv_map, std::string& text) {
    bool res = false;
    proto::KeyValueStorageData message;
    if (SerializeKvs(kv_map, message)) {
        res = google::protobuf::TextFormat::PrintToString(message, &text);
    }
    PER_LOG_INFO << "res :" << res;
    return res;
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
