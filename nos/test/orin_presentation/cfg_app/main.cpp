#include <iostream>
#include <memory>
#include <unistd.h>
#include <csignal>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "to_string.h"
#include "config_param.h"
#include "log/include/logging.h"

using namespace hozon::netaos::cfg;
bool g_stopFlag = false;
std::mutex mtx;
std::condition_variable cv;

const std::vector<uint8_t> DEFAULT_VIN = {0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30};
const std::vector<uint8_t> DEFAULT_CFG_WORD = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                               0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                               0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

void INTSigHandler(int32_t num)
{
    (void)num;
    g_stopFlag = true;
}


int main(int argc, char** argv)
{
    /*Need add SIGTERM from EM*/
    signal(SIGTERM, INTSigHandler);
    signal(SIGINT, INTSigHandler);
    signal(SIGPIPE, SIG_IGN);

    hozon::netaos::log::InitLogging(
        "CFG_APP",
        "CFG_APP",
        hozon::netaos::log::LogLevel::kError,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "/log",
        10,
        100
    );

    auto cfg_mgr = ConfigParam::Instance();
    cfg_mgr->Init(3000);
    std::vector<uint8_t> vin_number = DEFAULT_VIN;
    std::vector<uint8_t> cfg_word = DEFAULT_CFG_WORD;

    while (!g_stopFlag) {
        if (nullptr == cfg_mgr) {
            std::cout << "Error: cfg_mgr is nullptr!" << std::endl;
            break;
        }

        std::vector<uint8_t> data;
        cfg_mgr->GetParam<std::vector<uint8_t>>("dids/F190", data);
        if (17 == data.size()) {
            if (vin_number != data) {
                vin_number.assign(data.begin(), data.end());
                std::string vin = "";
                vin.assign(vin_number.begin(), vin_number.end());
                std::cout << "cfg monitor[VIN(ASCII)]: current VIN: " << vin << std::endl;
            }
        }
        else {
            if (vin_number != DEFAULT_VIN) {
                vin_number.assign(DEFAULT_VIN.begin(), DEFAULT_VIN.end());
                std::string vin = "";
                vin.assign(vin_number.begin(), vin_number.end());
                std::cout << "cfg monitor[VIN(ASCII)]: default VIN: : " << vin << std::endl;
            }
        }

        data.clear();
        cfg_mgr->GetParam<std::vector<uint8_t>>("dids/F170", data);
        if (58 == data.size()) {
            if (cfg_word != data) {
                cfg_word.assign(data.begin(), data.end());
                std::string cfgWord = UINT8_VEC_TO_STRING_DATA(cfg_word);
                std::cout << "cfg monitor[VechileCFG(HEX)]: current CfgWord: " << cfgWord << std::endl;
            }
        }
        else {
            if (cfg_word != DEFAULT_CFG_WORD) {
                cfg_word.assign(DEFAULT_CFG_WORD.begin(), DEFAULT_CFG_WORD.end());
                std::string cfgWord = UINT8_VEC_TO_STRING_DATA(cfg_word);
                std::cout << "cfg monitor[VechileCFG(HEX)]: default CfgWord: " << cfgWord << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    cfg_mgr->DeInit();
    return 0;
}
