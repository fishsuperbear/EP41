#include <iostream>
#include <fstream>  
#include <string>  
#include <sys/stat.h> 
#include <dirent.h>
#include <thread>
#include "gtest/gtest.h"
#include "log/include/logging.h"

using namespace hozon::netaos::log;

class LogUnitTest003:public ::testing::Test {

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

// isMain参数测试，最终日志输出以设置该flag为true的参数的为准
TEST_F(LogUnitTest003, LogIsMainTest_01) {
    PathRemove("LOG_TEST_003_1st");
    PathRemove("LOG_TEST_003_2nd");
    InitLogging("LOG_TEST_003_1st", "log unit test 003 1st", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 10, true);
    InitLogging("LOG_TEST_003_2nd", "log unit test 003 2nd", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 10);

    auto log = CreateLogger("LOGUNIT", "log unit test", LogLevel::kInfo);
    EXPECT_TRUE(log != nullptr);
    log->LogCritical() << "LogCritical";
    log->LogError() << "LogError";
    log->LogWarn() << "LogWarn";
    log->LogInfo() << "LogInfo";
    log->LogDebug() << "LogDebug";
    log->LogTrace() << "LogTrace";

    std::string directoryPath = "/opt/usr/log/soc_log/";
    std::string prefix1 = "LOG_TEST_003_1st";
    std::string prefix2 = "LOG_TEST_003_2nd";
    std::string suffix = ".log";
    std::string searchString_tarce = "LogTrace";
    std::string searchString_debug = "LogDebug";
    std::string searchString_info = "LogInfo";
    std::string searchString_error = "LogError";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    bool flag1 = false;  
    bool flag2 = false;  
    DIR* dir = opendir(directoryPath.c_str());
    if (dir != nullptr) {
        dirent* entry;
        // 遍历目录中的文件
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            std::string filePath = directoryPath + filename;
            // 判断前缀和后缀
            if (filename.substr(0, prefix1.length()) == prefix1 &&
                filename.substr(filename.length() - suffix.length()) == suffix) {
                flag1 = true;
            }
            if (filename.substr(0, prefix2.length()) == prefix2 &&
                filename.substr(filename.length() - suffix.length()) == suffix) {
                flag2 = true;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
    EXPECT_TRUE(flag1 == true);
    EXPECT_TRUE(flag2 == false);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}