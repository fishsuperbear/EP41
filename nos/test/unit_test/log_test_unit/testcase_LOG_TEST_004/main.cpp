#include <iostream>
#include <fstream>  
#include <string>  
#include <sys/stat.h> 
#include <dirent.h>
#include <thread>
#include "gtest/gtest.h"
#include "log/include/logging.h"

using namespace hozon::netaos::log;

class LogUnitTest004:public ::testing::Test {

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

// pureLogFormat参数测试，最终日志输出以纯净模式输出
TEST_F(LogUnitTest004, LogPureLogFormatTest) {
    PathRemove("LOG_TEST_004");
    InitLogging("LOG_TEST_004", "log unit test 004", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 10, true, true);

    auto log = CreateLogger("LOGUNIT", "log unit test", LogLevel::kInfo);
    EXPECT_TRUE(log != nullptr);
    log->LogCritical() << "LogCritical";
    log->LogError() << "LogError";
    log->LogWarn() << "LogWarn";
    log->LogInfo() << "LogInfo";
    log->LogDebug() << "LogDebug";
    log->LogTrace() << "LogTrace";

    std::string directoryPath = "/opt/usr/log/soc_log/";
    std::string prefix = "LOG_TEST_004";
    std::string suffix = ".log";
    std::string searchString = "LOGUNIT";
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
                EXPECT_TRUE(fileContent.find(searchString) == std::string::npos);
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