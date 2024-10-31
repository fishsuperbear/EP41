/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-07-05 14:06:57
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-12-16 15:40:44
 * @FilePath: /nos/service/config_server/include/cfg_manager.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 配置管理
 * Created on: Feb 7, 2023
 *
 */

#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_MANAGER_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_MANAGER_H_
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iostream>
#include <shared_mutex>
#include <string>

#include "include/cfg_data_def.h"
#include "include/cfg_logger.h"
#include "include/cfg_server_data_def.h"
#include "include/cfg_utils.h"
#include "include/file_storage.h"
#include "include/key_value_storage.h"
#include "include/per_error_domain.h"
#include "include/phm_client_instance.h"

namespace hozon {
namespace netaos {
namespace cfg {

class CfgManager {
 public:
    static CfgManager* Instance();
    ~CfgManager();
    void Init(std::string dir_path, std::string redundant_path, uint32_t maxcom_vallimit);
    void DeInit();
    bool ReadCfgFile(CfgServerData& datacfg);
    bool ReadKvsCfgFile(CfgServerData& datacfg, std::string paramName, uint8_t type);

    bool WriteCfgFile(CfgServerData& datacfg, std::string key);

 protected:
    std::string redundant_path_;
    std::string dir_path_;
    uint32_t maxcom_vallimit_;

 private:
    CfgManager();
    static CfgManager* sinstance_;
    std::string cfgfilename;
    std::string commondir = "configserver/";
};

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_MANAGER_H_
