#include <iostream>
#include <fstream>  
#include <string>  
#include <sys/stat.h> 
#include <dirent.h>
#include <thread>
#include "gtest/gtest.h"
#include <sys/statvfs.h>
#include "log/include/logging.h"

using namespace hozon::netaos::log;

class LogUnitTest006:public ::testing::Test {

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

bool hasEnoughSpace(const std::string& path, std::uintmax_t requiredSpace) {
    struct statvfs stat;
    if (statvfs(path.c_str(), &stat) != 0) {
        std::cerr << "Error checking space.\n";
        return false;
    }

    std::uintmax_t availableSpace = stat.f_bavail * stat.f_frsize;

    return availableSpace >= requiredSpace;
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

bool PathCreate(const std::string &pathName)
{
    bool bRet = false;

    std::string createPathCMD = "touch /opt/usr/log/soc_log/" + pathName;
    if (0 == system(createPathCMD.c_str())){
        bRet = true;
    }
    return bRet;
}
// 测试日志序号跳转逻辑
TEST_F(LogUnitTest006, LogRotateTest) {
    PathRemove("LOG_TEST_007");
    PathCreate("LOG_TEST_007_9995_2023-11-01_01-11-30.log");
    std::string directoryPath = "/opt/usr/log/soc_log/";

    std::uintmax_t requiredSpace = 30 * 1024 * 1024; 
    auto res = hasEnoughSpace(directoryPath, requiredSpace);
    ASSERT_TRUE(res == true);
    std::cout << "There is enough space.\n";

    InitLogging("LOG_TEST_007", "log unit test 004", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 5);

    auto log = CreateLogger("LOGUNIT", "log unit test", LogLevel::kInfo);
    ASSERT_TRUE(log != nullptr);
    std::string data(1950, 'A');
    int i = 0;
    while (i <= 7000)
    {
        log->LogCritical() << data;
        log->LogError() << data;
        log->LogWarn() << data;
        log->LogInfo() << data;
        i++;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::string prefix1 = "LOG_TEST_007_9996";
    std::string prefix2 = "LOG_TEST_007_9999";
    std::string prefix3 = "LOG_TEST_007_0001";
    std::string prefix4 = "LOG_TEST_007_0002";

    bool flag1 = false;
    bool flag2 = false;
    bool flag3 = false;
    bool flag4 = false;

    DIR* dir = opendir(directoryPath.c_str());
    if (dir != nullptr) {
        dirent* entry;
        // 遍历目录中的文件
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            std::string filePath = directoryPath + filename;
            // 判断前缀和后缀
            if (filename.substr(0, prefix1.length()) == prefix1) {
                flag1 = true;
            } else if (filename.substr(0, prefix2.length()) == prefix2) {
                flag2 = true;
            } else if (filename.substr(0, prefix3.length()) == prefix3) {
                flag3 = true;
            } else if (filename.substr(0, prefix4.length()) == prefix4) {
                flag4 = true;
            } else {
                // go on
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
    EXPECT_TRUE(flag1 == true);
    EXPECT_TRUE(flag2 == true);
    EXPECT_TRUE(flag3 == true);
    EXPECT_TRUE(flag4 == true);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}