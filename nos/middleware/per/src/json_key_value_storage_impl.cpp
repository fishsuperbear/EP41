/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: KV文件存储
 * Created on: Feb 7, 2023
 *
 */
#include "src/json_key_value_storage_impl.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>

#include "include/kvs_type.h"
#include "include/per_error_domain.h"
#include "include/per_utils.h"
#include "src/json_key_value_parser.h"

namespace hozon {
namespace netaos {
namespace per {

JsonKeyValueStorageImpl::JsonKeyValueStorageImpl(const StorageConfig& config) : _config(std::move(config)), recover(new FileRecovery()) {}

JsonKeyValueStorageImpl::~JsonKeyValueStorageImpl() { delete recover; }

hozon::netaos::core::Result<std::vector<std::string>> JsonKeyValueStorageImpl::GetAllKeys() const noexcept {
    std::vector<std::string> veclist;
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        veclist.push_back(kv.key);
    }
    PER_LOG_INFO << "veclist " << veclist.size();
    return hozon::netaos::core::Result<std::vector<std::string>>::FromValue(veclist);
}

hozon::netaos::core::Result<bool> JsonKeyValueStorageImpl::HasKey(const std::string& key) const noexcept {
    PER_LOG_INFO << "key " << key;
    bool haskey = false;
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            haskey = true;
            break;
        }
    }
    if (haskey) {
        return hozon::netaos::core::Result<bool>::FromValue(true);
    } else {
        return hozon::netaos::core::Result<bool>::FromValue(false);
    }
}

hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::RemoveKey(const std::string& key) noexcept {
    PER_LOG_INFO << "key " << key;
    for (std::vector<InnerKeyValue>::iterator it = kv_vec_.kv_vec.begin(); it != kv_vec_.kv_vec.end();) {
        if ((*it).key == key) {
            it = kv_vec_.kv_vec.erase(it);
            break;
        } else {
            ++it;
        }
    }
    return hozon::netaos::core::Result<void>::FromValue();
}

hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::RemoveAllKeys() noexcept {
    PER_LOG_INFO << "keysize " << kv_vec_.kv_vec.size();
    kv_vec_.kv_vec.clear();
    return hozon::netaos::core::Result<void>::FromValue();
}

hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SyncToStorage() noexcept {
    if (!PerUtils::CheckFreeSize(path_)) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kOutOfMemoryError);
    }
    std::string text;
    // Serialize kv_map into proto text.
    if (!JsonKeyValueParser::SerializeToJsonText(kv_vec_, text)) {
        PER_LOG_INFO << "SerializeToJsonText error";
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kSerializeError);
    }
    // open key value persistency file.
    std::ofstream ofs;
    ofs.open(path_, std::ios_base::binary | std::ios_base::out);
    if (!ofs.is_open()) {
        PER_LOG_INFO << "is_open error";
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kPhysicalStorageError);
    }
    // write key value serialized content into file.
    if (!ofs.write(text.data(), text.size())) {
        PER_LOG_INFO << "write error";
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kWriteAccessError);
    }
    if (!ofs.flush()) {
        PER_LOG_INFO << "flush error";
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kWriteAccessError);
    }
    bool res = recover->BackUpHandle(path_, _config);
    PER_LOG_INFO << "BackUpHandle " << res;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<int>(const std::string& key, const int& value) noexcept {
    std::string valstr = PerUtils::NumToString<int>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_INT;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_INT;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: int"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<uint64_t>(const std::string& key, const uint64_t& value) noexcept {
    std::string valstr = PerUtils::NumToString<uint64_t>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_UINT64;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_UINT64;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: uint64_t"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<bool>(const std::string& key, const bool& value) noexcept {
    std::string valstr = PerUtils::NumToString<bool>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_BOOL;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_BOOL;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: bool"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<float>(const std::string& key, const float& value) noexcept {
    std::string valstr = PerUtils::NumToString<float>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_FLOAT;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_FLOAT;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: float"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<double>(const std::string& key, const double& value) noexcept {
    std::string valstr = PerUtils::NumToString<double>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_DOUBLE;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_DOUBLE;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: double"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<std::string>(const std::string& key, const std::string& value) noexcept {
    std::string valstr = value;
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_STRING;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_STRING;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: string"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<google::protobuf::Message>(const std::string& key, const google::protobuf::Message& message) noexcept {
    std::string valstr;
    // message.SerializeToString(&value);
    google::protobuf::TextFormat::PrintToString(message, &valstr);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_NESTED_MESSAGE;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_NESTED_MESSAGE;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: google::protobuf::Message"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<std::vector<int>>(const std::string& key, const std::vector<int>& value) noexcept {
    std::string valstr = PerUtils::ChopLineStringEx<int>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_VEC_INT;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_VEC_INT;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: std::vector<int>"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<std::vector<bool>>(const std::string& key, const std::vector<bool>& value) noexcept {
    std::string valstr = PerUtils::ChopLineStringEx<bool>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_VEC_BOOL;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_VEC_BOOL;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: std::vector<bool>"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<std::vector<float>>(const std::string& key, const std::vector<float>& value) noexcept {
    std::string valstr = PerUtils::ChopLineStringEx<float>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_VEC_FLOAT;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_VEC_FLOAT;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: std::vector<float>"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<std::vector<double>>(const std::string& key, const std::vector<double>& value) noexcept {
    std::string valstr = PerUtils::ChopLineStringEx<double>(value);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_VEC_DOUBLE;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_VEC_DOUBLE;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: std::vector<double>"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<std::vector<std::string>>(const std::string& key, const std::vector<std::string>& value) noexcept {
    std::vector<uint8_t> uint8vec;
    PerUtils::VecToBytes(value, uint8vec);
    std::vector<uint16_t> vec;
    for (size_t size = 0; size < uint8vec.size(); size++) {
        vec.push_back(uint8vec[size]);
    }
    std::string valstr = PerUtils::ChopLineStringEx<uint16_t>(vec);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_VEC_STRING;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_VEC_STRING;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: std::vector<string>"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::SetBaseValue<std::vector<uint8_t>>(const std::string& key, const std::vector<uint8_t>& value) noexcept {
    std::vector<uint16_t> vec;
    for (size_t size = 0; size < value.size(); size++) {
        vec.push_back(value[size]);
    }
    std::string valstr = PerUtils::ChopLineStringEx<uint16_t>(vec);
    bool haskey = false;
    for (InnerKeyValue& kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            kv.value.type = JSON_PER_TYPE_VEC_UINT8;
            kv.value.string = valstr;
            haskey = true;
            break;
        }
    }
    if (!haskey) {
        InnerKeyValue kv;
        kv.key = key;
        kv.value.type = JSON_PER_TYPE_VEC_UINT8;
        kv.value.string = valstr;
        kv_vec_.kv_vec.push_back(kv);
    }
    PER_LOG_INFO << "key: " << key << " type: std::vector<uint8_t>"
                 << " valuesize: " << valstr.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

// void JsonKeyValueStorageImpl::BcdStringToVector(const std::string& input, std::vector<uint8_t>& ouput) {
//     uint32_t len = input.size() >> 1;
//     uint8_t* buff = new uint8_t[len];
//     memset(buff, 0x00, len);
//     for (uint32_t index = 0; index < len; ++index) {
//         uint8_t left = (uint8_t)(input[index << 1]), right = (uint8_t)(input[(index << 1) + 1]);
//         left = (left >= '0' && left <= '9') ? (left - '0') : (left - 'a' + 10);
//         right = (right >= '0' && right <= '9') ? (right - '0') : (right - 'a' + 10);
//         buff[index] = (uint8_t)(left << 4) | right;
//     }
//     ouput.assign(buff, buff + len);
//     delete[] buff;
// }
// void JsonKeyValueStorageImpl::VectorToBcdString(const std::vector<uint8_t>& input, std::string& ouput) {
//     uint32_t len = input.size();
//     char* buff = new char[len << 1];
//     memset(buff, 0x00, len);
//     for (size_t j = 0; j < len; j++) {
//         uint8_t high = (input[j] >> 4) & 0x0F, low = input[j] & 0x0F;
//         buff[j << 1] = (high < 10) ? ('0' + high) : ('a' + high - 10);
//         buff[(j << 1) + 1] = (low < 10) ? ('0' + low) : ('a' + low - 10);
//     }
//     ouput.assign(buff, len << 1);
//     delete[] buff;
// }

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<int>(const std::string& key, int& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_INT) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::stringToNum<int>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type: int"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}
template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<uint64_t>(const std::string& key, uint64_t& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_UINT64) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::stringToNum<uint64_t>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type: uint64_t"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}
template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<bool>(const std::string& key, bool& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_BOOL) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::stringToNum<bool>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type: bool"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<float>(const std::string& key, float& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_FLOAT) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::stringToNum<float>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type: float"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<double>(const std::string& key, double& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_DOUBLE) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::stringToNum<double>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type: double"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<std::string>(const std::string& key, std::string& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_STRING) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                value = kv.value.string;
                PER_LOG_INFO << "key: " << key << " type: std::string"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<google::protobuf::Message>(const std::string& key, google::protobuf::Message& message) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_NESTED_MESSAGE) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                google::protobuf::TextFormat::ParseFromString(kv.value.string, &message);
                PER_LOG_INFO << "key: " << key << " type: google::protobuf::Message"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<std::vector<int>>(const std::string& key, std::vector<int>& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_VEC_INT) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::ChopStringLineEx<int>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type:  std::vector<int>"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<std::vector<bool>>(const std::string& key, std::vector<bool>& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_VEC_BOOL) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::ChopStringLineEx<bool>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type:  std::vector<bool>"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<std::vector<float>>(const std::string& key, std::vector<float>& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_VEC_FLOAT) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::ChopStringLineEx<float>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type:  std::vector<float>"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<std::vector<double>>(const std::string& key, std::vector<double>& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_VEC_DOUBLE) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                PerUtils::ChopStringLineEx<double>(kv.value.string, value);
                PER_LOG_INFO << "key: " << key << " type:  std::vector<double>"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<std::vector<std::string>>(const std::string& key, std::vector<std::string>& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_VEC_STRING) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                std::vector<uint8_t> uint8vec;
                std::vector<uint16_t> val;
                PerUtils::ChopStringLineEx<uint16_t>(kv.value.string, val);
                for (size_t size = 0; size < val.size(); size++) {
                    uint8vec.push_back(val[size]);
                }
                PerUtils::BytesToVec(uint8vec, value);
                PER_LOG_INFO << "key: " << key << " type:  std::vector<std::string>"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}
template <>
hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::GetBaseValue<std::vector<uint8_t>>(const std::string& key, std::vector<uint8_t>& value) const noexcept {
    for (InnerKeyValue kv : kv_vec_.kv_vec) {
        if (kv.key == key) {
            if (kv.value.type != JSON_PER_TYPE_VEC_UINT8) {
                return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
            } else {
                std::vector<uint16_t> val;
                PerUtils::ChopStringLineEx<uint16_t>(kv.value.string, val);
                for (size_t size = 0; size < val.size(); size++) {
                    value.push_back(val[size]);
                }
                PER_LOG_INFO << "key: " << key << " type:  std::vector<uint8_t>"
                             << " valuesize: " << kv.value.string.size();
            }
            return hozon::netaos::core::Result<void>::FromValue();
        }
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

hozon::netaos::core::Result<void> JsonKeyValueStorageImpl::Open(const std::string& path) {
    if (!recover->RecoverHandle(path, _config)) {
        PER_LOG_INFO << "RecoverHandle  false";
    }
    path_ = path;
    std::ifstream ifs;
    // open key value persistency file.
    ifs.open(path_, std::ios_base::binary | std::ios_base::in);
    if (!ifs.is_open()) {
        PER_LOG_INFO << " is_open false  " << path_;
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kPhysicalStorageError);
    }
    // Read key value file all content once.
    size_t fsize = ifs.seekg(0, std::ios_base::end).tellg();
    PER_LOG_INFO << " seekg fsize   " << fsize;
    ifs.seekg(0, std::ios_base::beg);
    std::vector<char> buf;
    buf.resize(fsize);
    if (!ifs.read(buf.data(), buf.size())) {
        PER_LOG_INFO << "read: error";
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kReadAccessError);
    }
    std::string data(buf.data(), buf.size());
    if (!JsonKeyValueParser::ParseJsonText(data, kv_vec_)) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kDeSerializeError);
    }
    return hozon::netaos::core::Result<void>::FromValue();
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
