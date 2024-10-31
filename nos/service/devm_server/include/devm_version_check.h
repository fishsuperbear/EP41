/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: devm version check
 */

#ifndef DEVM_SERVER_VERSION_CHECK_H_
#define DEVM_SERVER_VERSION_CHECK_H_

#include <string>
#include "cfg/include/config_param.h"
#include "zmq_ipc/manager/zmq_ipc_client.h"


using namespace hozon::netaos::cfg;
namespace hozon {
namespace netaos {
namespace devm_server {


class DevmVerCheck {
public:
    DevmVerCheck();
    ~DevmVerCheck();
    void Run();
    void SetStopFlag();

private:
    bool ReadMajorAndSWTVersionFromFile(std::string& major_v, std::string& swt_v);
    std::string ReadSocVersionFromFile();
    std::string ReadDsvVersionFromFile();
    bool ReadDidsFromFile(std::map<std::string, std::string>& dids_value);

    int32_t ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data);
    bool ReadDidsFromCfg(std::string& value, const std::string& dids);
    bool WriteDidsToCfg(const std::string& value, const std::string& dids);
    std::string GetUpgradeStatus();

    std::string major_version_{};
    std::string soc_version_dyna_{};
    std::string mcu_version_dyna_{};
    std::string dsv_version_dyna_{};
    std::string swt_version_{};
    std::string swt_version_dyna_{};
    ConfigParam* cfg_mgr_{};

    std::shared_ptr<hozon::netaos::zmqipc::ZmqIpcClient> client_;

    bool stop_flag_{false};
};


}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
#endif  // DEVM_SERVER_VERSION_CHECK_H_


