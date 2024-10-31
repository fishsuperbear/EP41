/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Store the config indifferent between client and server.
 * Create: 2020-06-03
 */

#ifndef RTF_COM_CONFIG_ROLE_CONFIG_H
#define RTF_COM_CONFIG_ROLE_CONFIG_H
#include <set>
#include <map>
#include <string>
#include "rtf/com/types/ros_types.h"

namespace rtf {
namespace com {
namespace utils{
class Logger;
}
namespace config {
class DDSRoleConfig {
public:
    using Trans = TransportMode;
    /**
     * @brief DDSRoleConfig constructor
     */
    DDSRoleConfig();

    /**
     * @brief DDSEntityConfig constructor
     *
     * @param[in] transportModes transport modes of the entity include client and server
     */
    explicit DDSRoleConfig(const std::set<TransportMode>& transportModes);

    /**
     * @brief DDSRoleConfig deconstructor
     */
    virtual ~DDSRoleConfig() = default;

    /**
     * @brief Set entity transport mode
     *
     * @param[in] transportModes transport modes of the entity
     * @param[in] role           choose transport mode role config
     */
    void SetTransportMode(const std::set<TransportMode>& transportModes, const Role& role) noexcept;

    /**
     * @brief Get entity transport mode
     *
     * @param[in] role         choose transportMode role config
     * @return std::set<TransportMode> transport mode of this role, default both will return total transport mode config
     */
    std::set<TransportMode> GetTransportMode(const Role& role) const noexcept;

    /**
     * @brief Get a role type
     *
     * @return Role type set.
     */
    std::set<rtf::com::Role> GetRoleType() const noexcept;
private:
    std::map<Role, std::set<TransportMode>> transportModeMap_ = {
        {Role::SERVER, {TransportMode::UDP, TransportMode::SHM}},
        {Role::CLIENT, {TransportMode::UDP, TransportMode::SHM}}
    };
    std::set<rtf::com::Role> roleTypeSet_;
    std::shared_ptr<rtf::com::utils::Logger> logger_;
};
}
}
}
#endif
