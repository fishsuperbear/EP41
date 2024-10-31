/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Authentication.hpp
 */

#ifndef DDS_CORE_POLICY_AUTHENTICATION_HPP
#define DDS_CORE_POLICY_AUTHENTICATION_HPP

#include <cstdint>
#include <string>

namespace dds {
namespace core {
namespace policy {
/* String max length is 65, "UID(max 32 byte)" + ":" + "GID(max 32 bytes) */
constexpr uint16_t MAX_SHM_FILE_OWNER_LENGTH = 65U;
/**
 * @brief Configures Qos authentication
 * shmFileOwner is a string in the format of "[UID][:GID]"
 * It is usually recommended to only use usernames that begin
 * with a lower case letter or an underscore, followed by lower case letters,
 * digits, underscores, or dashes. They can end with a dollar sign.
 * In regular expression terms: [a-z_][a-z0-9_-]*[$]?
 */
class Authentication {
public:
    Authentication(void) = default;
    ~Authentication(void) = default;

    bool ShmFileOwner(std::string owner);

    const std::string &ShmFileOwner(void) const noexcept
    {
        return shmFileOwner_;
    }
private:
    std::string shmFileOwner_;
};
}
}
}

#endif /* DDS_CORE_POLICY_AUTHENTICATION_HPP */
