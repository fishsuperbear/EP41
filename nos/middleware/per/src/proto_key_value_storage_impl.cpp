/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: KV文件存储
 * Created on: Feb 7, 2023
 *
 */
#include "src/proto_key_value_storage_impl.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "include/kvs_type.h"
#include "include/per_error_domain.h"
#include "include/per_utils.h"
#include "src/proto_key_value_parser.h"

namespace hozon {
namespace netaos {
namespace per {

ProtoKeyValueStorageImpl::ProtoKeyValueStorageImpl(const StorageConfig& config) : _config(std::move(config)), recover(new FileRecovery()) {}

ProtoKeyValueStorageImpl::~ProtoKeyValueStorageImpl() { delete recover; }

hozon::netaos::core::Result<std::vector<std::string>> ProtoKeyValueStorageImpl::GetAllKeys() const noexcept {
    // std::vector<std::string> veclist;
    // for (auto& it : kv_map_) {
    //     veclist.push_back(it.first);
    // }
    std::vector<std::string> veclist(kv_map_.size());
    std::transform(kv_map_.begin(), kv_map_.end(), veclist.begin(), [](const std::pair<std::string, InnerValue>& pair) { return pair.first; });
    PER_LOG_INFO << "veclist " << veclist.size();
    return hozon::netaos::core::Result<std::vector<std::string>>::FromValue(veclist);
}

hozon::netaos::core::Result<bool> ProtoKeyValueStorageImpl::HasKey(const std::string& key) const noexcept {
    PER_LOG_INFO << "key " << key;
    if (kv_map_.count(key)) {
        return hozon::netaos::core::Result<bool>::FromValue(true);
    } else {
        return hozon::netaos::core::Result<bool>::FromValue(false);
    }
}

hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::RemoveKey(const std::string& key) noexcept {
    PER_LOG_INFO << "key " << key;
    kv_map_.erase(key);
    return hozon::netaos::core::Result<void>::FromValue();
}

hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::RemoveAllKeys() noexcept {
    PER_LOG_INFO << "keysize " << kv_map_.size();
    kv_map_.clear();
    return hozon::netaos::core::Result<void>::FromValue();
}

hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SyncToStorage() noexcept {
    if (!PerUtils::CheckFreeSize(path_)) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kOutOfMemoryError);
    }
    std::string text;
    // Serialize kv_map into proto text.
    if (!ProtoKeyValueParser::SerializeToProtoText(kv_map_, text)) {
        PER_LOG_INFO << "SerializeToProtoText error";
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
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<int>(const std::string& key, const int& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_INT) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_INT;
    kv_map_[key].buf.resize(sizeof(int32_t));
    *reinterpret_cast<int32_t*>(kv_map_[key].buf.data()) = value;
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " value: " << value;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<uint64_t>(const std::string& key, const uint64_t& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_UINT64) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_UINT64;
    kv_map_[key].buf.resize(sizeof(uint64_t));
    *reinterpret_cast<uint64_t*>(kv_map_[key].buf.data()) = value;
    PER_LOG_INFO << "key: " << key << " type: " << (uint64_t)kv_map_[key].value_type << " value: " << value;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<bool>(const std::string& key, const bool& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_BOOL) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_BOOL;
    kv_map_[key].buf.resize(sizeof(bool));
    *reinterpret_cast<bool*>(kv_map_[key].buf.data()) = value;
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " value: " << value;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<float>(const std::string& key, const float& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_FLOAT) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_FLOAT;
    kv_map_[key].buf.resize(sizeof(float));
    *reinterpret_cast<float*>(kv_map_[key].buf.data()) = value;
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " value: " << value;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<double>(const std::string& key, const double& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_DOUBLE) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_DOUBLE;
    kv_map_[key].buf.resize(sizeof(double));
    *reinterpret_cast<double*>(kv_map_[key].buf.data()) = value;
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " value: " << value;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<std::string>(const std::string& key, const std::string& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_STRING) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_STRING;
    kv_map_[key].buf.resize(value.size());
    ::memcpy(kv_map_[key].buf.data(), value.data(), value.size());
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " value: " << value;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<google::protobuf::Message>(const std::string& key, const google::protobuf::Message& message) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_NESTED_MESSAGE) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_NESTED_MESSAGE;
    std::string value;
    // message.SerializeToString(&value);
    google::protobuf::TextFormat::PrintToString(message, &value);
    kv_map_[key].buf.resize(value.size());
    ::memcpy(kv_map_[key].buf.data(), value.data(), value.size());
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type;
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<std::vector<int>>(const std::string& key, const std::vector<int>& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_VEC_INT) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_VEC_INT;
    kv_map_[key].int32buf.clear();
    for (size_t index = 0; index < value.size(); index++) {
        kv_map_[key].int32buf.push_back(value[index]);
    }
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " valuesize: " << value.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<std::vector<bool>>(const std::string& key, const std::vector<bool>& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_VEC_BOOL) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_VEC_BOOL;
    kv_map_[key].boolbuf.clear();
    for (size_t index = 0; index < value.size(); index++) {
        kv_map_[key].boolbuf.push_back(value[index]);
    }
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " valuesize: " << value.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<std::vector<float>>(const std::string& key, const std::vector<float>& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_VEC_FLOAT) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_VEC_FLOAT;
    kv_map_[key].floatbuf.clear();
    for (size_t index = 0; index < value.size(); index++) {
        kv_map_[key].floatbuf.push_back(value[index]);
    }
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " valuesize: " << value.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<std::vector<double>>(const std::string& key, const std::vector<double>& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_VEC_DOUBLE) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_VEC_DOUBLE;
    kv_map_[key].doublebuf.clear();
    for (size_t index = 0; index < value.size(); index++) {
        kv_map_[key].doublebuf.push_back(value[index]);
    }
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " valuesize: " << value.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<std::vector<std::string>>(const std::string& key, const std::vector<std::string>& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_VEC_STRING) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_VEC_STRING;
    kv_map_[key].stringbuf.clear();
    for (size_t index = 0; index < value.size(); index++) {
        kv_map_[key].stringbuf.push_back(value[index]);
    }
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " valuesize: " << value.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::SetBaseValue<std::vector<uint8_t>>(const std::string& key, const std::vector<uint8_t>& value) noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        if (kv_map_[key].value_type != PROTO_PER_TYPE_STRING) {
            return hozon::netaos::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }
    kv_map_[key].value_type = PROTO_PER_TYPE_STRING;
    kv_map_[key].buf.resize(value.size());
    ::memcpy(kv_map_[key].buf.data(), value.data(), value.size());
    PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_[key].value_type << " valuesize: " << value.size();
    return hozon::netaos::core::Result<void>::FromValue();
}

// void ProtoKeyValueStorageImpl::BcdStringToVector(const std::string& input, std::vector<uint8_t>& ouput) {
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
// void ProtoKeyValueStorageImpl::VectorToBcdString(const std::vector<uint8_t>& input, std::string& ouput) {
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
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<int>(const std::string& key, int& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        value = *reinterpret_cast<const int32_t*>(kv_map_.at(key).buf.data());
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " value: " << value;
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}
template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<uint64_t>(const std::string& key, uint64_t& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        value = *reinterpret_cast<const uint64_t*>(kv_map_.at(key).buf.data());
        PER_LOG_INFO << "key: " << key << " type: " << (uint64_t)kv_map_.at(key).value_type << " value: " << value;
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}
template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<bool>(const std::string& key, bool& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        value = *reinterpret_cast<const bool*>(kv_map_.at(key).buf.data());
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " value: " << value;
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<float>(const std::string& key, float& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        value = *reinterpret_cast<const float*>(kv_map_.at(key).buf.data());
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " value: " << value;
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<double>(const std::string& key, double& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        value = *reinterpret_cast<const double*>(kv_map_.at(key).buf.data());
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " value: " << value;
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<std::string>(const std::string& key, std::string& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        value = std::string(reinterpret_cast<const char*>(kv_map_.at(key).buf.data()), kv_map_.at(key).buf.size());
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " value: " << value;
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<google::protobuf::Message>(const std::string& key, google::protobuf::Message& message) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        std::string valstr = std::string(reinterpret_cast<const char*>(kv_map_.at(key).buf.data()), kv_map_.at(key).buf.size());
        google::protobuf::TextFormat::ParseFromString(valstr, &message);
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type;
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<std::vector<int>>(const std::string& key, std::vector<int>& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        for (size_t index = 0; index < kv_map_.at(key).int32buf.size(); index++) {
            value.push_back(kv_map_.at(key).int32buf[index]);
        }
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " valuesize: " << value.size();
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<std::vector<bool>>(const std::string& key, std::vector<bool>& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        for (size_t index = 0; index < kv_map_.at(key).boolbuf.size(); index++) {
            value.push_back(kv_map_.at(key).boolbuf[index]);
        }
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " valuesize: " << value.size();
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<std::vector<float>>(const std::string& key, std::vector<float>& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        for (size_t index = 0; index < kv_map_.at(key).floatbuf.size(); index++) {
            value.push_back(kv_map_.at(key).floatbuf[index]);
        }
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " valuesize: " << value.size();
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<std::vector<double>>(const std::string& key, std::vector<double>& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        for (size_t index = 0; index < kv_map_.at(key).doublebuf.size(); index++) {
            value.push_back(kv_map_.at(key).doublebuf[index]);
        }
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " valuesize: " << value.size();
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<std::vector<std::string>>(const std::string& key, std::vector<std::string>& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        for (size_t index = 0; index < kv_map_.at(key).stringbuf.size(); index++) {
            value.push_back(kv_map_.at(key).stringbuf[index]);
        }
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " valuesize: " << value.size();
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}
template <>
hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::GetBaseValue<std::vector<uint8_t>>(const std::string& key, std::vector<uint8_t>& value) const noexcept {
    if (kv_map_.find(key) != kv_map_.end()) {
        for (size_t index = 0; index < kv_map_.at(key).buf.size(); index++) {
            value.push_back(kv_map_.at(key).buf[index]);
        }
        PER_LOG_INFO << "key: " << key << " type: " << (int32_t)kv_map_.at(key).value_type << " valuesize: " << value.size();
        return hozon::netaos::core::Result<void>::FromValue();
    }
    return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
}

hozon::netaos::core::Result<void> ProtoKeyValueStorageImpl::Open(const std::string& path) {
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
    // Parse proto text.
    std::string data(buf.data(), buf.size());
    // ParseProtoText(const std::string& data, InnerKeyValueMap& key_value_map)
    if (!ProtoKeyValueParser::ParseProtoText(data, kv_map_)) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kDeSerializeError);
    }
    return hozon::netaos::core::Result<void>::FromValue();
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
