#include <iostream>
#include <memory>
#include "cm/include/method.h"
#include "chassis_ota_method.h"
#include "chassis_ota_methodPubSubTypes.h"
#include "default_logger.h"

int main(int argc, char** argv) {
    DefaultLogger::GetInstance().InitLogger();
    hozon::netaos::cm::Client<ChassisOtaMethod, ChassisOtaMethod>
         client(std::make_shared<ChassisOtaMethodPubSubType>(),
             std::make_shared<ChassisOtaMethodPubSubType>());
    if(client.Init(0, "/soc/chassis_ota_method")) {
        std::cout << "method client init fail." << std::endl;
        return -1;
    }
    
    int online = client.WaitServiceOnline(5 * 1000);
    if(online) {
        std::cout << "Service Online fail , return: " << online << std::endl;
        return -1;
    }
    

    std::shared_ptr<ChassisOtaMethod> req_data = std::make_shared<ChassisOtaMethod>();
    std::shared_ptr<ChassisOtaMethod> resp_data = std::make_shared<ChassisOtaMethod>();
    req_data->fire_forget(false);
    client.Request(req_data, resp_data, 500);

    std::cout << "BDCS1_PowerManageMode: " << static_cast<int>(resp_data->BDCS1_PowerManageMode()) << std::endl;

    client.Deinit();

    return 0;
}