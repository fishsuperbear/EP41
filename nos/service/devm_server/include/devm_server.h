#ifndef DEVM_SERVER_H_
#define DEVM_SERVER_H_

#include <memory>
#include "devm_udp_mcu_version.h"
#include "devm_udp_temp_vol.h"
#include "devm_version_check.h"
#include "devm_server_impl_zmq.h"

namespace hozon {
namespace netaos {
namespace devm_server {

class DevmServerImpl;
class DevmServer {
   public:
    DevmServer();
    ~DevmServer();

    void Init();
    void Run();
    void DeInit();

   private:
    //std::shared_ptr<DevmServerImpl> devm_server_impl_;
    std::shared_ptr<DevmUdpMcuVersion> devm_udp_mcu_version_;
    std::thread udp_mcu_version_thread_;
    std::shared_ptr<DevmUdpTempAndVol> devm_udp_temp_vol_;
    std::thread udp_temp_vol_thread_;
    std::shared_ptr<DevmVerCheck> devm_ver_check_;
    std::thread ver_check_thread_;
    std::shared_ptr<DevmServerImplZmq> devm_server_zmq_;
};

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon

#endif  //DEVM_SERVER_H_