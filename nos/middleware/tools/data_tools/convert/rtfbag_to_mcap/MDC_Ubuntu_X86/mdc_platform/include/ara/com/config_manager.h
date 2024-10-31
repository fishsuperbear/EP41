/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: This provide the read function of the ConfigManager.
 * Create: 2022-05-22
 */

#ifndef ARA_COM_CONFIG_MANAGER_H
#define ARA_COM_CONFIG_MANAGER_H

#include "ara/core/result.h"

namespace ara {
namespace com {

class ConfigManager {
public:
    virtual ~ConfigManager() =default;
    static std::shared_ptr<ConfigManager>& GetInstance();
    virtual ara::core::Result<void> AddConfigDir(const ara::core::String& configDir) = 0;

protected:
    ConfigManager() = default;
};

}
}


#endif //ARA_COM_CONFIG_MANAGER_H
