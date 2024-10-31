/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: KeyValueStorage class header
 *              This class is an abstraction of data types
 * Create: 2019-06-23
 */

#ifndef ARA_GODEL_COMMON_KVS_KEY_VALUE_STORAGE_H
#define ARA_GODEL_COMMON_KVS_KEY_VALUE_STORAGE_H

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ara    {
namespace godel  {
namespace common {
namespace kvs    {
class KeyValueStorage;
using KvsObject = KeyValueStorage;
using KvsArray = std::vector<KvsObject>;

class KeyValueStorage {
public:
    KeyValueStorage(void) : key_(""), type_(0U)
    {
    }

    KeyValueStorage(KeyValueStorage const &other) = default;

    KeyValueStorage(KeyValueStorage && other) = default;

    ~KeyValueStorage(void) = default;

    KeyValueStorage& operator=(KeyValueStorage const &other) & = default;

    KeyValueStorage& operator=(KeyValueStorage && other) & = default;

    bool operator==(KeyValueStorage const &other) const noexcept;

    bool operator!=(KeyValueStorage const &other) const noexcept;

    std::string const &GetKey(void) const noexcept;

    template <typename T> bool GetValue(T& value) const noexcept
    {
        if ((!IsTypeMatched(value)) || (value_ == nullptr)) {
            return false;
        }

        value = *(std::static_pointer_cast<T>(value_));
        return true;
    }

    bool GetValue(std::string const &key, KvsObject& value) const;

    void SetKey(std::string const &key);

    template <typename T> void SetValue(T const &value) noexcept
    {
        type_  = typeid(T).hash_code();
        value_ = std::static_pointer_cast<void>(std::make_shared<T>(value));
    };

    bool GetUInt(uint32_t& val) const noexcept;

    bool GetInt(int32_t& val) const noexcept;

    bool GetDouble(double& val) const noexcept;
private:
    std::string           key_;
    size_t                type_; // the return type of type_info.hashcode() is size_t
    std::shared_ptr<void> value_;

    template <typename T> bool IsTypeMatched(T const &value) const noexcept
    {
        if (type_ != typeid(T).hash_code()) {
            return false;
        }
        static_cast<void>(value);
        return true;
    }
};
} // namespace kvs
} // namespace common
} // namespace godel
} // namespace ara
#endif // ARA_GODEL_COMMON_KVS_KEY_VALUE_STORAGE_H_ */
