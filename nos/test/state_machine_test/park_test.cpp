#include <unistd.h>
#include "config_param.h"
#include "log/include/default_logger.h"



int main() {
    DefaultLogger::GetInstance().InitLogger();
    hozon::netaos::cfg::CfgResultCode initres = hozon::netaos::cfg::ConfigParam::Instance()->Init();
    if (initres == 0) {
        DF_LOG_INFO << "Init error:";
        return 0;
    }
    sleep(1);

    uint8_t g_ivalue_mode_req = 2;
    uint8_t g_ivalue_fm_req = 1;
    DF_LOG_INFO << "g_ivalue_mode_req Set =" << g_ivalue_mode_req;
    hozon::netaos::cfg::ConfigParam::Instance()->SetParam<uint8_t>("system/mode_req", g_ivalue_mode_req);
    hozon::netaos::cfg::ConfigParam::Instance()->SetParam<uint8_t>("system/fm_stat", g_ivalue_fm_req);

    int switchMode_Counter = 0;
    uint8_t g_ivalue_running_mode = 0;
    //判断是否超时，计时周期为200毫秒
    while (switchMode_Counter++ < 10)
    {
        hozon::netaos::cfg::ConfigParam::Instance()->GetParam<uint8_t>("system/running_mode", g_ivalue_running_mode);            
        DF_LOG_INFO << "g_ivalue_running_mode Get =" << g_ivalue_running_mode;

        //可以切入泊车模式
        if (g_ivalue_running_mode == 2)
        {
            DF_LOG_INFO << "change park state success......";
        }
        else
        {
            DF_LOG_INFO << "change park state fail.";
        }
        sleep(1);
    }

    hozon::netaos::cfg::ConfigParam::Instance()->DeInit();
    DF_LOG_INFO << "Deinit end.";
    return 0;
}