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

class LogUnitTest005:public ::testing::Test {

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

// 日志数量测试，产生的日志文件数量，小于等于设置值
TEST_F(LogUnitTest005, LogMaxFileNumTest) {
    PathRemove("LOG_TEST_005");
    std::string directoryPath = "/opt/usr/log/soc_log/";

    std::uintmax_t requiredSpace = 30 * 1024 * 1024; 
    auto res = hasEnoughSpace(directoryPath, requiredSpace);
    ASSERT_TRUE(res == true);
    std::cout << "There is enough space.\n";

    InitLogging("LOG_TEST_005", "log unit test 004", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 5, 5);

    auto log = CreateLogger("LOGUNIT", "log unit test", LogLevel::kInfo);
    EXPECT_TRUE(log != nullptr);

    std::string data(1950, 'A');

    int i = 0;
    while (i <= 8000)
    {
        log->LogError() << data;
        log->LogInfo() << data;
        i++;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::string prefix = "LOG_TEST_005";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    uint16_t count = 0;
    DIR* dir = opendir(directoryPath.c_str());
    if (dir != nullptr) {
        dirent* entry;
        // 遍历目录中的文件
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            std::string filePath = directoryPath + filename;
            if (filename.substr(0, prefix.length()) == prefix) {
                count ++;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
    std::cout << "log file count is : " << count << std::endl;
    EXPECT_TRUE(count <= 5);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}