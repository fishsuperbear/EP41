#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "cfg_logger.h"
#include "config_param.h"

sig_atomic_t g_stopFlag = 0;
using namespace hozon::netaos::cfg;
using namespace std;

void SigHandler(int signum) {
    g_stopFlag = 1;
    CONFIG_LOG_INFO << "--- cfg SigHandler enter, signum [" << signum << "] ---";
}

void uint8func(const string& clientname, const string& key, const uint8_t& value) {
    CONFIG_LOG_INFO << "uint8func func receive the event that the value of param:test"
                    << " is set to " << clientname << "  " << key << "  " << (int32_t)value << "  ";
}
void uint8func1(const string& clientname, const string& key, const uint8_t& value) {
    CONFIG_LOG_INFO << "uint8func1 func1 receive the event that the value of param:test"
                    << " is set to " << clientname << "  " << key << "  " << (int32_t)value << "  ";
}
void uint8func2(const string& clientname, const string& key, const uint8_t& value) {
    CONFIG_LOG_INFO << "uint8func2 func2 receive the event that the value of param:test"
                    << " is set to " << clientname << "  " << key << "  " << (int32_t)value << "  ";
}
void vecstringfunc3(const string& clientname, const string& key, const std::vector<string>& value) {
    CONFIG_LOG_INFO << "vecstringfunc3 receive the event that the value of param:test"
                    << " is set to " << clientname << "  " << key;
    for (size_t size = 0; size < value.size(); size++) {
        CONFIG_LOG_INFO << " " << value[size];
    }
}
void doublefunc(const string& clientname, const string& key, const double& value) {
    CONFIG_LOG_INFO << "doublefunc func2 receive the event that the value of param:test"
                    << " is set to " << clientname << "  " << key << "  " << value << "  ";
}
CfgResultCode ResponseParamfunc(const int32_t& value) {
    CONFIG_LOG_INFO << "ResponseParamfunc func2 receive the event that the value of param:test"
                    << " is set to " << value;
    return CfgResultCode(1);  // 1： 参数处理流程成功； 2： 参数处理流程失败
}

// export LD_LIBRARY_PATH=./output/x86_2004/lib:$LD_LIBRARY_PATH

int main(int argc, char* argv[]) {
    signal(SIGTERM, SigHandler);
    signal(SIGINT, SigHandler);
    signal(SIGPIPE, SIG_IGN);
    std::string key = "test";
    std::string mode;
#ifdef BUILD_FOR_MDC
    mode = "./";
#elif BUILD_FOR_ORIN
    mode = "/log/";
#else
    mode = "./";
#endif
    hozon::netaos::log::InitLogging("CFG_TEST",                           // the id of application
                                    "cfg_test",                           // the log id of application
                                    hozon::netaos::log::LogLevel::kInfo,  // the log level of application
                                    hozon::netaos::log::HZ_LOG2CONSOLE,   // the output log mode
                                    mode,                                 // the log file directory, active when output log to file
                                    10,                                   // the max number log file , active when output log to file
                                    20                                    // the max size of each  log file , active when output log to file
    );
    auto cfgMgr = ConfigParam::Instance();
    CfgResultCode initres = cfgMgr->Init(3000);
    if (initres == 0) {
        CONFIG_LOG_INFO << "Init error:";
        return 0;
    }
    uint8_t res;
    cfgMgr->MonitorParam<uint8_t>("test", uint8func);
    cfgMgr->MonitorParam<uint8_t>("test", uint8func1);
    cfgMgr->MonitorParam<uint8_t>("test1", uint8func2);
    cfgMgr->MonitorParam<double>("testdouble", doublefunc);
    cfgMgr->MonitorParam<std::vector<string>>("testvecstring", vecstringfunc3);
    cfgMgr->MonitorParam<std::vector<string>>("testvecstring11", vecstringfunc3);
    vector<std::string> clients;
    res = cfgMgr->GetMonitorClients("test", clients);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " :" << clients.size();
    for (size_t size = 0; size < clients.size(); size++) {
        CONFIG_LOG_INFO << "GetMonitorClients res:" << clients[size].c_str();
    }
    uint8_t val = 0;
    cfgMgr->SetParam<uint8_t>("test", val, CONFIG_SYNC_PERSIST);
    uint8_t val1 = 1;
    res = cfgMgr->GetParam<uint8_t>("test", val1);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " test:" << (uint32_t)val1;
    bool val111 = 1;
    res = cfgMgr->GetParam<bool>("test", val111);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " test:" << (uint32_t)val111;

    cfgMgr->SetParam<uint8_t>("", val, CONFIG_SYNC_PERSIST);

    uint8_t val2 = 2;
    cfgMgr->SetDefaultParam<uint8_t>("test", val2);
    uint8_t val3 = 3;
    res = cfgMgr->GetParam<uint8_t>("test", val3);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " test:" << (uint32_t)val3;

    uint8_t val4 = 4;
    cfgMgr->SetParam<uint8_t>("test", val4, CONFIG_SYNC_PERSIST);
    uint8_t val5 = 5;
    res = cfgMgr->GetParam<uint8_t>("test", val5);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " test:" << (int32_t)val5;
    cfgMgr->ResetParam("test");
    cfgMgr->ResetParam("test1");
    cfgMgr->ResetParam("test2");
    uint8_t val6 = 5;
    res = cfgMgr->GetParam<uint8_t>("test", val6);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " test:" << (int32_t)val6;
    cfgMgr->DelParam("test");
    cfgMgr->DelParam("test2");
    uint8_t val7 = 7;
    res = cfgMgr->GetParam<uint8_t>("test", val7);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " test:" << (int32_t)val7;
    int32_t val8 = 8;
    cfgMgr->SetParam<int32_t>("testint32", val8, CONFIG_SYNC_PERSIST);
    int32_t val9 = 9;
    res = cfgMgr->GetParam<int32_t>("testint32", val9);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testint32:" << val9;

    double val10 = 10.1;
    cfgMgr->SetParam<double>("testdouble", val10, CONFIG_SYNC_PERSIST);
    double val11 = 11.1;
    res = cfgMgr->GetParam<double>("testdouble", val11);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testdouble:" << val11;

    double val101 = 11.0;
    cfgMgr->SetParam<double>("testdouble", val101, CONFIG_SYNC_PERSIST);

    float val12 = 12.2;
    cfgMgr->SetParam<float>("testfloat", val12, CONFIG_SYNC_PERSIST);
    float val13 = 13.3;
    res = cfgMgr->GetParam<float>("testfloat", val13);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testfloat:" << val13;

    std::string val14 = "14.0";
    cfgMgr->SetParam<std::string>("teststring", val14, CONFIG_SYNC_PERSIST);
    std::string val15 = "15.1";
    res = cfgMgr->GetParam<std::string>("teststring", val15);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " teststring:" << val15;

    bool val16 = true;
    cfgMgr->SetParam<bool>("testbool", val16, CONFIG_SYNC_PERSIST);
    bool val17 = false;
    res = cfgMgr->GetParam<bool>("testbool", val17);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testbool:" << val17;

    long val18 = 18;
    cfgMgr->SetParam<long>("testlong", val18, CONFIG_SYNC_PERSIST);
    long val19 = 19;
    res = cfgMgr->GetParam<long>("testlong", val19);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testlong:" << val19;

    size_t vecsize = 5;
    vector<bool> val20;
    for (size_t size = 0; size < vecsize; size++) {
        val20.push_back(size % 2);
    }
    cfgMgr->SetParam<vector<bool>>("testvecbool", val20, CONFIG_SYNC_PERSIST);
    vector<bool> val21;
    res = cfgMgr->GetParam<vector<bool>>("testvecbool", val21);
    for (size_t size = 0; size < val21.size(); size++) {
        CONFIG_LOG_INFO << " " << val21[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecboolsize:" << val21.size();

    vector<int32_t> val24;
    for (size_t size = 0; size < vecsize; size++) {
        val24.push_back(size);
    }
    cfgMgr->SetParam<vector<int32_t>>("testvecint32", val24, CONFIG_SYNC_PERSIST);
    vector<int32_t> val25;
    res = cfgMgr->GetParam<vector<int32_t>>("testvecint32", val25);
    for (size_t size = 0; size < val25.size(); size++) {
        CONFIG_LOG_INFO << " " << val25[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecint32size:" << val25.size();

    vector<float> val26;
    for (size_t size = 0; size < vecsize; size++) {
        val26.push_back(size * 1.1);
    }
    cfgMgr->SetParam<vector<float>>("testvecfloat", val26, CONFIG_SYNC_PERSIST);
    vector<float> val27;
    res = cfgMgr->GetParam<vector<float>>("testvecfloat", val27);
    for (size_t size = 0; size < val27.size(); size++) {
        CONFIG_LOG_INFO << " " << val27[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecfloatsize:" << val27.size();

    vector<double> val28;
    for (size_t size = 0; size < vecsize; size++) {
        val28.push_back(size * 1.1);
    }
    cfgMgr->SetParam<vector<double>>("testvecdouble", val28, CONFIG_SYNC_PERSIST);
    vector<double> val29;
    res = cfgMgr->GetParam<vector<double>>("testvecdouble", val29);
    for (size_t size = 0; size < val29.size(); size++) {
        CONFIG_LOG_INFO << " " << val29[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecdoublesize:" << val29.size();

    vector<long> val30;
    for (size_t size = 0; size < vecsize; size++) {
        val30.push_back(size);
    }
    cfgMgr->SetParam<vector<long>>("testveclong", val30, CONFIG_SYNC_PERSIST);
    vector<long> val31;
    res = cfgMgr->GetParam<vector<long>>("testveclong", val31);
    for (size_t size = 0; size < val31.size(); size++) {
        CONFIG_LOG_INFO << " " << val31[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testveclongsize:" << val31.size();

    vector<uint8_t> val22;
    for (size_t size = 0; size < vecsize; size++) {
        val22.push_back(size);
    }
    cfgMgr->SetParam<vector<uint8_t>>("testvecuint8", val22, CONFIG_SYNC_PERSIST);
    vector<uint8_t> val23;
    res = cfgMgr->GetParam<vector<uint8_t>>("testvecuint8", val23);
    for (size_t size = 0; size < val23.size(); size++) {
        CONFIG_LOG_INFO << " " << (uint32_t)val23[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecuint8size:" << val23.size();

    vector<std::string> val32;
    for (size_t size = 0; size < vecsize; size++) {
        val32.push_back("testvecstring00000000" + std::to_string(size));
    }
    cfgMgr->SetParam<vector<std::string>>("testvecstring", val32);
    vector<std::string> val33;
    res = cfgMgr->GetParam<vector<std::string>>("testvecstring", val33);
    for (size_t size = 0; size < val33.size(); size++) {
        CONFIG_LOG_INFO << " " << val33[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecstringsize:" << val33.size();

    vector<std::string> val34;
    for (size_t size = 0; size < vecsize; size++) {
        val34.push_back("testvecstring1" + std::to_string(size * size * size) + "000 001");
    }
    cfgMgr->SetParam<vector<std::string>>("testvecstring11", val34, CONFIG_SYNC_PERSIST);
    vector<std::string> val35;
    res = cfgMgr->GetParam<vector<std::string>>("testvecstring11", val35);
    for (size_t size = 0; size < val35.size(); size++) {
        CONFIG_LOG_INFO << " " << val35[size];
    }
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecstring11:" << val35.size();

    float val36 = 36.6;
    cfgMgr->SetDefaultParam<float>("testfloat", val36);
    float val37 = 37.7;
    res = cfgMgr->GetParam<float>("testfloat", val37);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testfloat:" << val37;

    float val381 = 88.8;
    cfgMgr->SetParam<float>("testfloat1", val381, CONFIG_ASYNC_PERSIST);
    float val361 = 316.6;
    cfgMgr->SetDefaultParam<float>("testfloat1", val361);
    float val3811 = 99.8;
    cfgMgr->SetParam<float>("testfloat1", val3811, CONFIG_ASYNC_PERSIST);
    float val38 = 38.8;
    cfgMgr->SetParam<float>("testfloat", val38, CONFIG_SYNC_PERSIST);
    float val39 = 39.9;
    res = cfgMgr->GetParam<float>("testfloat", val39);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testfloat:" << val39;
    cfgMgr->ResetParam("testfloat");
    float val40 = 40.0;
    res = cfgMgr->GetParam<float>("testfloat", val40);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testfloat:" << val40;
    int32_t va41 = 0;
    cfgMgr->SetParam<int32_t>("testint32_t", va41);
    int32_t val42 = 1;
    res = cfgMgr->GetParam<int32_t>("testint32_t", val42);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testint32_t:" << val42;
    long va43 = 0;
    cfgMgr->SetParam<long>("testlong", va43);
    long val44 = 1;
    res = cfgMgr->GetParam<long>("testlong", val44);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testlong:" << val44;

    cfgMgr->ResponseParam<int32_t>("RequestParam", ResponseParamfunc);
    CONFIG_LOG_INFO << "RequestParam begin:";
    int32_t val45 = 44;
    cfgMgr->RequestParam<int32_t>("RequestParam", val45);
    CONFIG_LOG_INFO << "RequestParam end:";

    int32_t val46 = 1;
    res = cfgMgr->GetParam<int32_t>("RequestParam", val46);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " RequestParam:" << val46;
    vector<uint8_t> val47;
    for (size_t size = 0; size < 25 * 1024 * 1024; size++) {
        val47.push_back(size);
    }
    cfgMgr->SetParam<vector<uint8_t>>("testvecuint8large", val47);
    vector<uint8_t> val48;
    for (size_t size = 0; size < 5 * 1024; size++) {
        val48.push_back(size);
    }
    cfgMgr->SetParam<vector<uint8_t>>("testvecuint8filestoragesync", val48, CONFIG_SYNC_PERSIST);
    val48.clear();
    res = cfgMgr->GetParam<vector<uint8_t>>("testvecuint8filestoragesync", val48);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecuint8filestoragesync:" << val48.size();
    val48.clear();
    for (size_t size = 0; size < 15 * 1024; size++) {
        val48.push_back(size);
    }
    cfgMgr->SetParam<vector<uint8_t>>("testvecuint8filestorageasync", val48, CONFIG_ASYNC_PERSIST);
    val48.clear();
    res = cfgMgr->GetParam<vector<uint8_t>>("testvecuint8filestorageasync", val48);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " testvecuint8filestorageasync:" << val48.size();
    val48.clear();
    for (size_t size = 0; size < 3 * 1024; size++) {
        val48.push_back(size);
    }
    cfgMgr->SetDefaultParam<vector<uint8_t>>("testvecuint8filestoragedefault", val48);
    val48.clear();
    for (size_t size = 0; size < 6 * 1024; size++) {
        val48.push_back(size);
    }

    cfgMgr->SetParam<vector<uint8_t>>("hozon/testvecuint8filestoragedir", val48, CONFIG_ASYNC_PERSIST);
    val48.clear();
    res = cfgMgr->GetParam<vector<uint8_t>>("hozon/testvecuint8filestoragedir", val48);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " hozon/testvecuint8filestoragedir:" << val48.size();
    val48.clear();
    for (size_t size = 0; size < 30; size++) {
        val48.push_back(size);
    }
    for (size_t size = 0; size < 5; size++) {
        std::string file = "hozon/testvecuint8filestoragedir" + std::to_string(size);
        cfgMgr->SetParam<vector<uint8_t>>(file, val48, CONFIG_SYNC_PERSIST);
    }

    for (size_t size = 0; size < 5; size++) {
        std::string file = "test/testvecuint8filestoragedir" + std::to_string(size);
        std::string val = "test" + std::to_string(size);
        cfgMgr->SetParam<std::string>(file, val, CONFIG_SYNC_PERSIST);
    }
    std::string jsonval;
    res = cfgMgr->GetParam<std::string>("test/all", jsonval);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " jsonval:  " << jsonval.c_str();
    std::string file1(13 * 1024, 'a');
    cfgMgr->SetParam<std::string>("largefile", file1, CONFIG_SYNC_PERSIST);
    std::string data;
    res = cfgMgr->GetParam<std::string>("largefile", data);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << " largefile " << data.size();
    std::string file = "calibration/aaa";
    cfgMgr->SetParam<std::string>(file, file, CONFIG_SYNC_PERSIST);
    std::string data1;
    res = cfgMgr->GetParam<std::string>(file, data1);
    CONFIG_LOG_INFO << "GetParam res:" << (int32_t)res << file << " :" << data1.size();

    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    cfgMgr->UnMonitorParam("RequestParam");
    cfgMgr->UnMonitorParam("test");
    cfgMgr->UnMonitorParam("hozon");
    cfgMgr->UnMonitorParam("test1");
    cfgMgr->UnMonitorParam("testdouble");
    cfgMgr->UnMonitorParam("testvecstring");
    cfgMgr->UnMonitorParam("testvecstring11");
    cfgMgr->UnMonitorParam("testvecstring11");
    cfgMgr->UnMonitorParam("testvecstring111");

    cfgMgr->DeInit();
    return 0;
}