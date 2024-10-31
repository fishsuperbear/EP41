#include <iostream>
#include <fstream>  
#include <string>  
#include <sys/stat.h> 
#include <dirent.h>
#include <thread>
#include "gtest/gtest.h"
#include "log/include/logging.h"

using namespace hozon::netaos::log;

class LogUnitTest002:public ::testing::Test {

protected:
    static void SetUpTestSuite() {
        std::cout << "=== SetUpTestSuite ===" << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "=== TearDownTestSuite ===" << std::endl;
    }

    void SetUp() override {
        system("export LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH");
    }

    void TearDown() override {}

protected:
};

bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

void deleteFile(const std::string& filename) {
    if (std::remove(filename.c_str()) != 0) {
        std::cerr << "Error deleting file: " << filename << std::endl;
    }
}

bool PathRemove(const std::string &appId)
{
    bool bRet = false;
    std::string rmCMD = "rm -r  /opt/usr/log/soc_log/" + appId + "*";
    if (0 == system(rmCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

// 测试使用json文件初始化日志
TEST_F(LogUnitTest002, LogInitWithJsonTest) {
    PathRemove("LOG_TEST_002");
    std::string filePath = "../conf/log_cfg.json";
    std::ifstream fileStream(filePath);
    ASSERT_TRUE(fileStream.good() == true);
    InitLogging(filePath);
    auto log = CreateLogger("LOGUNIT002", "log unit test", LogLevel::kInfo);
    EXPECT_TRUE(log != nullptr);

    log->LogCritical() << "LogCritical";
    log->LogError() << "LogError";
    log->LogWarn() << "LogWarn";
    log->LogInfo() << "LogInfo";
    log->LogDebug() << "LogDebug";
    log->LogTrace() << "LogTrace";

    std::string prefix = "LOG_TEST_002";
    std::string suffix = ".log";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::string directoryPath = "/opt/usr/log/soc_log/";
    bool flag = false;  
    DIR* dir = opendir(directoryPath.c_str());
    if (dir != nullptr) {
        dirent* entry;
        // 遍历目录中的文件
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            std::string filePath = directoryPath + filename;
            // 判断前缀和后缀
            if (filename.substr(0, prefix.length()) == prefix &&
                filename.substr(filename.length() - suffix.length()) == suffix) {
                flag = true;
                break;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
    EXPECT_TRUE(flag == true);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}