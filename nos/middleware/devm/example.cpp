
#include "include/devm_device_info.h"
#include "include/common/devm_logger.h"
#include <iostream>


using namespace hozon::netaos::devm;
using namespace hozon::netaos::devm_server;
int main()
{
    DevmServerLogger::GetInstance().InitLogging();

    DevmClientDeviceInfo device_info;
    // std::vector<uint8_t> vinData = {0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x30,0x31,0x32,0x33,0x34,0x35,0x36};
    // device_info.WriteVinData(vinData);
    // std::vector<uint8_t> snData = {0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39};
    // device_info.WriteTesterSNData(snData);


    std::string vin;
    device_info.GetVinNumber(vin);
    std::cout << "vin:" << vin <<std::endl;
    vin.clear();
    device_info.GetEcuSerialNumber(vin);
    std::cout << "sn :" << vin <<std::endl;

    return 0;
}

