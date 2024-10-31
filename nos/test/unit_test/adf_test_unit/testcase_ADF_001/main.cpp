#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <signal.h>
#include <unistd.h>
#include <memory>
#include <signal.h>
#include <fcntl.h>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include "gtest/gtest.h"
#include "adf/include/log.h"
#include "diag/libsttask/STLogDef.h"
#include "yaml-cpp/yaml.h"

using namespace std;

#define CLIENT_SUCC             (SIGRTMIN + 1)
#define COMMON_RECV_SUCC        (SIGRTMIN + 2)

std::string main_folder_path_;
int main_pid_ = 0;

class DebugLogger {
public:
    DebugLogger(bool need_log) :
        _need_log(need_log) {

    }

    ~DebugLogger() {
        if (_need_log) {
            std::cout << std::endl;
        }
    }

    template<typename T>
    DebugLogger& operator<<(const T& value) {
        if (_need_log) {
            std::cout << _head << value;
        }
        
        return *this;
    }

private:
    bool _need_log = false;
    std::string _head;
};

#define DEBUG_LOG           DebugLogger(true)
#define HEAD_LOG            DebugLogger(true) << "\033[32m[==========]\033[0m "
#define SEP_LOG             DebugLogger(true) << "\033[32m[----------]\033[0m "
#define RUN_LOG             DebugLogger(true) << "\033[32m[ RUN      ]\033[0m "
#define OK_LOG              DebugLogger(true) << "\033[32m[       OK ]\033[0m "
#define FAIL_LOG            DebugLogger(true) << "\033[31m[  FAILED  ]\033[0m "
#define PASS_LOG            DebugLogger(true) << "\033[32m[  PASSED  ]\033[0m "

void RedirectLog() {
    int fd = open("/dev/null", O_WRONLY);
    dup2(fd, 1);  // redirect stdout
}

int CreateClientProcess(const std::string &folder_path, const std::string& module, int parent_pid, char** evn) {
    std::string bin_path = folder_path + std::string("/node_test_recv");
    DEBUG_LOG << "node_test_recv path:" << bin_path << " module: " << module;
    int ret = fork();
    if (ret < 0) {
        DEBUG_LOG << "fail to fork " << stderr;
    }
    else if (ret != 0) {
        // parent process
        DEBUG_LOG << "start client " << module;
        sleep(1);
        return ret;
    } 
    else if (ret == 0) {
        RedirectLog();
        execle(bin_path.c_str(), 
              bin_path.c_str(),
                module.c_str(), 
                std::to_string(parent_pid).c_str(), 
                NULL,
                evn);
    }
    
    return -1;
}

int CreateCommSendProcess(const std::string &folder_path, const std::string& module, int parent_pid, char** evn) {
    std::string bin_path = folder_path + std::string("/node_test_send");
    DEBUG_LOG << "node_test_send path:" << bin_path << ". module: " << module;
    int ret = fork();
    if (ret < 0) {
        DEBUG_LOG << "fail to fork";
    }
    else if (ret != 0) {
        // parent process
        DEBUG_LOG << "start common send " << module;
        sleep(1);
        return ret;
    } 
    else if (ret == 0) {
        RedirectLog();
        DEBUG_LOG << "evn[0]: " << evn[0] << ", evn[1]: " << evn[1];
        execle(bin_path.c_str(), 
              bin_path.c_str(),
            "send", 
            module.c_str(), 
            std::to_string(parent_pid).c_str(), 
            NULL,
            evn);
    }
    return -1;
}
//A c function to get your own executable's execution directory. 
// Sometimes you want to know the program source 

#ifndef MAX_PATH
#define MAX_PATH 256
#endif

char *get_process_path () {
    static char buffer[MAX_PATH];
    int len;
    if ((len = readlink ("/proc/self/exe", buffer, sizeof (buffer) - 1)) != -1) {
        buffer[len] = '\0';
    }
    else
        *buffer = '\0';
    return (buffer);
}

class AdfTest : public ::testing::Test{
protected:
    static void SetUpTestSuite(){
        std::cout << "=== SetUpTestSuite ===" << std::endl;
    }
	static void TearDownTestSuite(){
        cout << "=== TearDownTestSuite ===" << endl;
    } 
    void SetUp() override {
	}

	void TearDown() override {
	}
};
bool send_succ = false;
bool recv_succ = false;

void SignalHandler(int32_t sig) {
    if (sig == CLIENT_SUCC) {
        recv_succ = true;
        DEBUG_LOG << "client succ";
    }
    else if (sig == COMMON_RECV_SUCC) {
        send_succ = true;
        DEBUG_LOG << "common recv succ";
    }
}

bool VerifyModule(const std::string &folder_path, const std::string& module, int pid) {
    bool result = false;

    send_succ = false;
    recv_succ = false;
    // std::string adf_config_path = folder_path + std::string("/../conf/adf_test_001.yaml");
    // YAML::Node config_node = YAML::LoadFile(adf_config_path);
    // if (!config_node) {
    //     DEBUG_LOG << "Fail to load config file ";
    //     return false;
    // }
    // std::string ld_path = config_node["libPath"].as<std::string>();
    std::string ld_path = std::string("/app/lib");
    // char* ld_path = std::getenv("LD_LIBRARY_PATH");
    DEBUG_LOG << "getenv LD_LIBRARY_PATH: " << ld_path;
    std::ostringstream oss; 
    oss << "LD_LIBRARY_PATH=" << ld_path;
    DEBUG_LOG << "ld_path: " << oss.str();
    std::string ld_evn = oss.str();
    char* evn[2] = {const_cast<char*>(ld_evn.c_str()), NULL};

    int client_pid = CreateClientProcess(folder_path, module, pid, evn);
    // DEBUG_LOG << "Create common send process test." << client_pid;
    int comm_send_pid = CreateCommSendProcess(folder_path, module, pid, evn);

    const int max_time_sec = 10;
    int i = 0;
    while (i < max_time_sec) {
        if (recv_succ == true) {
            DEBUG_LOG << module << " succ";
            result = true;
            break;
        }
        sleep(1);
        ++i;
    }

    DEBUG_LOG << "going to stop";
    if(client_pid > 0) {
        kill(client_pid, SIGTERM);
        waitpid(client_pid, nullptr, 0);
        DEBUG_LOG << "client joined";
    }
    if(comm_send_pid > 0) {
        kill(comm_send_pid, SIGTERM);
        waitpid(comm_send_pid, nullptr, 0);
        DEBUG_LOG << "send joined";
    }
    return result;
}
TEST_F(AdfTest, StartTestIdl) {
    std::cout << "test adf idl" << std::endl;   
    bool result = VerifyModule(main_folder_path_, "chassis", main_pid_);
	
	EXPECT_TRUE(result);
}

TEST_F(AdfTest, StartTestProto) {
    std::cout << "test adf proto" << std::endl;
    bool result = VerifyModule(main_folder_path_, "proto", main_pid_);
    EXPECT_TRUE(result);
}

int main(int argc, char* argv[]) {
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc, argv);

    main_pid_ = getpid();
    signal(CLIENT_SUCC, SignalHandler);
    signal(COMMON_RECV_SUCC, SignalHandler);
    std::cout << "get_path_to_me: " << std::string(get_process_path()) << std::endl;
    std::string binary_path = std::string(get_process_path());
    std::size_t pos = 0;
    for (std::size_t i = 0; i < binary_path.size(); ++i) {
        if (binary_path[i] == '/') {
            pos = i;
        }
    }
    main_folder_path_ = binary_path.substr(0, pos);
    std::cout << "folder path: " << main_folder_path_ << std::endl;
    return  RUN_ALL_TESTS();
}