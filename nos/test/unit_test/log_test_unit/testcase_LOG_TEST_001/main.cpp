#include <iostream>
#include <fstream>  
#include <string>  
#include <sys/stat.h> 
#include <dirent.h>
#include <thread>
#include "gtest/gtest.h"
#include "log/include/logging.h"

using namespace hozon::netaos::log;

class LogUnitTest001:public ::testing::Test {

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

std::string readFileToString(const std::string& filePath) {  

    std::ifstream file(filePath);  
    if (!file.is_open()) {  
        std::cerr << "Failed to open file: " << filePath << std::endl;  
        return "";  
    }  
    std::string content((std::istreambuf_iterator<char>(file)),  

                        std::istreambuf_iterator<char>());  
    file.close();  
    return content;  
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

// 测试不同日志等级的日志的输出情况
TEST_F(LogUnitTest001, LogLevelTest) {
    PathRemove("LOG_TEST_001");
    InitLogging("LOG_TEST_001", "log unit test", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 10);
    auto log = CreateLogger("LOGUNIT", "log unit test", LogLevel::kInfo);
    EXPECT_TRUE(log != nullptr);
    log->LogCritical() << "LogCritical";
    log->LogError() << "LogError";
    log->LogWarn() << "LogWarn";
    log->LogInfo() << "LogInfo";
    log->LogDebug() << "LogDebug";
    log->LogTrace() << "LogTrace";

    std::string directoryPath = "/opt/usr/log/soc_log/";
    std::string prefix = "LOG_TEST_001";
    std::string suffix = ".log";
    std::string searchString_tarce = "LogTrace";
    std::string searchString_debug = "LogDebug";
    std::string searchString_info = "LogInfo";
    std::string searchString_error = "LogError";
    std::this_thread::sleep_for(std::chrono::seconds(1));
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
                // 文件存在，读取内容
                std::string fileContent = readFileToString(filePath);
                std::cout << "file str: " << fileContent << std::endl;
                // 判断字符串C是否在文件内容中
                EXPECT_TRUE(fileContent.find(searchString_tarce) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString_debug) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString_info) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString_error) != std::string::npos);
                deleteFile(filePath);
                break;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}