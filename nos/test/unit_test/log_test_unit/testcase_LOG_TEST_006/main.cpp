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

bool isFileSizeExceeded(const std::string& directory, const std::string& prefix, std::uintmax_t maxSize) {
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        DIR* dir = opendir(directory.c_str());
        if (!dir) {
            std::cerr << "Error opening directory.\n";
            return false;
        }

        std::uintmax_t maxFileSize = 0;

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_REG && strncmp(entry->d_name, prefix.c_str(), prefix.size()) == 0) {
                std::string filePath = directory + "/" + entry->d_name;

                struct stat fileStat;
                if (stat(filePath.c_str(), &fileStat) == 0) {
                    std::uintmax_t fileSize = fileStat.st_size;
                    if (fileSize > maxFileSize) {
                        maxFileSize = fileSize;
                    }
                } else {
                    std::cerr << "Error getting file size for: " << filePath << "\n";
                }
            }
        }

        closedir(dir);

        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);
        std::chrono::seconds maxTimeout(10);
        if (elapsed >= maxTimeout) {
            std::cout << "Timeout reached. Exiting.\n";
            return false;
        }

        if (maxFileSize > maxSize) {
            std::cout << "File size exceeded: " << maxFileSize << " bytes.\n";
            return true;
        }
    }

    return false;
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

// 日志最大size，产生的日志文件数量，小于等于设置值
TEST_F(LogUnitTest006, LogMaxFileSizeTest) {
    PathRemove("LOG_TEST_006");
    std::string directoryPath = "/opt/usr/log/soc_log/";

    std::uintmax_t requiredSpace = 12 * 1024 * 1024; 
    auto res = hasEnoughSpace(directoryPath, requiredSpace);
    ASSERT_TRUE(res == true);
    std::cout << "There is enough space.\n";

    std::thread([&]{
        InitLogging("LOG_TEST_006", "log unit test 004", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 10);

        auto log = CreateLogger("LOGUNIT", "log unit test", LogLevel::kInfo);
        EXPECT_TRUE(log != nullptr);
        int i = 0;
        std::string data(1950, 'A');
        while (i <= 5200)
        {
            log->LogError() << data;
            i++;
        }
    }).detach();
    
    std::string prefix = "LOG_TEST_006_0001";

    std::uintmax_t maxFileSize = 10 * 1024 * 1024;  // 10MB
    // 30s 超时
    auto resEx = isFileSizeExceeded(directoryPath, prefix, maxFileSize);
    if (resEx) {
        std::cout << "File size exceeded.\n";
    } else {
        std::cout << "File size is within limits.\n";
    }
    EXPECT_TRUE(resEx == false);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}