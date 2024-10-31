#include <iostream>
#include <fstream>
#include <unistd.h>
#include "system_monitor_info.h"
#include "middleware/cm/include/skeleton.h"
#include "middleware/cfg/include/config_param.h"
#include "service/idl/generated/monitorPubSubTypes.h"

using namespace hozon::netaos::cfg;
using namespace hozon::netaos::cm;

// help
void printHelp()
{
    std::cout << R"(
        用法:
            stmm [监控类型]                输出当前记录的最新监控结果

        参数:
            help        帮助信息
            all         所有监控类型
            cpu         cpu监控
            mem         mem监控
            disk        disk监控
            file        文件监控
            mnand       emmc和ufs监控
            network     网络监控
            process     非ap进程监控
            temp        温度监控
            voltage     电压监控
    )" << std::endl;
}

// control help
void printControlHelp()
{
    std::cout << R"(
        用法:
            stmm [id] [type] [value]               控制监控参数

        id:
            0         cpu监控
            1         mem监控
            2         disk监控
            3         文件监控
            4         emmc和ufs监控
            5         网络监控
            6         非ap进程监控
            7         温度监控
            8         电压监控
            9         所有监控

        type:
            0         监控开关控制(on: 开启, off: 关闭)
            1         监控周期控制(大于0的值, 单位ms)
            2         文件记录周期控制(0: 不记录文件 大于0的值记录文件, 单位s)
            3         文件记录路径控制(给定一个绝对路径)
            4         是否告警控制(1: 告警 0: 不告警)
            5         告警值控制(大于0的值)
            6         后处理开关控制(on: 开启, off: 关闭)

        example:
            stmm 0 0 on             打开cpu监控
            stmm 0 1 3000           设置cpu监控周期为3000ms一次
            stmm 0 2 3              设置cpu监控文件落盘周期为3s一次
            stmm 0 3 /log/          设置cpu监控文件落盘路径为/log/
            stmm 0 4 1              打开cpu监控告警
            stmm 0 5 90             设置cpu告警值为90也就是cpu使用率大于90后会进行告警
            stmm 0 6 on             打开cpu监控后处理
    )" << std::endl;
}

void MointorInfo(const std::string& type)
{
    std::string result = "";
    if ("help" == type) {
        printHelp();
    }
    else if ("helpc" == type) {
        printControlHelp();
    }
    else if ("all" == type) {
        for (auto& item : MONITOR_RECORD_FILE_PATH_MAP) {
            std::string info = SystemMonitorInfo::getInstance()->GetMonitorInfo(item.first);
            if ("" == info) {
                result += SystemMonitorInfo::getInstance()->GetMonitorInfoFromFile(item.first);
            }
            else {
                result += info;
            }
        }
    }
    else {
        auto itr = MONITOR_RECORD_FILE_PATH_MAP.find(type);
        if (MONITOR_RECORD_FILE_PATH_MAP.end() == itr) {
            std::cout << "Parameter input error, refer to the following help: " << std::endl;
            printHelp();
        }
        else {
            std::string info = SystemMonitorInfo::getInstance()->GetMonitorInfo(type);
            if ("" == info) {
                result += SystemMonitorInfo::getInstance()->GetMonitorInfoFromFile(type);
            }
            else {
                result += info;
            }
        }
    }

    if ("" != result) {
        std::cout << result;
    }
}

void MointorControl(const std::string& id, const std::string& type, const std::string& value)
{
    // get vin number
    std::string vin = "";
    ConfigParam::Instance()->Init(1000);
    ConfigParam::Instance()->GetParam<std::string>("dids/F190", vin);

    std::shared_ptr<monitor_control_eventPubSubType> controlPubsubtype = std::make_shared<monitor_control_eventPubSubType>();
    Skeleton control = Skeleton(controlPubsubtype);
    control.Init(0, "monitor_control_eventTopic_" + vin);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!control.IsMatched()) {
        std::cout << "Unable to send control message due to not matching." << std::endl;
        return;
    }

    uint controlId = std::strtoul(id.c_str(), 0, 0);
    uint controlType = std::strtoul(type.c_str(), 0, 0);
    if (!((controlId >= 0 && controlId <= 8) && (controlType >= 0 && controlType <= 5))) {
        std::cout << "Parameter input error, refer to the following help: " << std::endl;
        printControlHelp();
        return;
    }

    std::shared_ptr<monitor_control_event> controlData = std::make_shared<monitor_control_event>();
    controlData->monitor_id(controlId);
    controlData->control_type(controlType);
    controlData->control_value(value);
    if (0 == control.Write(controlData)) {
        std::cout << "Send monitor control event message succeed." << std::endl;
    }
    else {
        std::cout << "Send monitor control event message failed." << std::endl;
    }

    control.Deinit();
    ConfigParam::Instance()->DeInit();
}

int main(int argc, char * argv[])
{
    SystemMonitorInfo::getInstance()->Init();
    switch(argc)
    {
        case 2:
            // monitor info
            MointorInfo(std::string(argv[1]));
            break;
        case 4:
            // monitor control
            MointorControl(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
            break;
        default:
            std::cout << "Please check argv numbers, refer to the following help: " << std::endl;
            printHelp();
            break;
    }

    SystemMonitorInfo::getInstance()->DeInit();
    return 0;
}