#include <iostream>
#include <fstream>  
#include <string>  
#include <sys/stat.h> 
#include <dirent.h>
#include <thread>
#include "gtest/gtest.h"
#include "log/include/logging.h"

using namespace hozon::netaos::log;

class LogUnitTest009:public ::testing::Test {

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

bool PathRemove(const std::string &appId)
{
    bool bRet = false;
    std::string rmCMD = "rm -r  /opt/usr/log/soc_log/" + appId + "*";
    if (0 == system(rmCMD.c_str())){
        bRet = true;
    }
    return bRet;
}

// 输出一些OP日志，包括 ctx (OP_01 OP_02 OP_03)
TEST_F(LogUnitTest009, LogOpTest01) {
    PathRemove("LOG_TEST_009_1");
    PathRemove("LOG_TEST_009_2");
    PathRemove("OP_01");
    PathRemove("OP_02");
    PathRemove("OP_03");
    PathRemove("OP_04");

    InitLogging("LOG_TEST_009_1", "log unit test 009", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 10);

    auto log1 = CreateOperationLogger("OP_01", "op log unit test", LogLevel::kDebug);
    EXPECT_TRUE(log1 != nullptr);
    log1->LogInfo() << "LogInfo 1.1";
    log1->LogDebug() << "LogDebug 1.1";
    log1->LogError() << "LogError 1.1";

    auto log2 = CreateOperationLogger("OP_02", "op log unit test", LogLevel::kDebug);
    EXPECT_TRUE(log2 != nullptr);
    log2->LogInfo() << "LogInfo 1.2";
    log2->LogDebug() << "LogDebug 1.2";
    log2->LogError() << "LogError 1.2";

    auto log3 = CreateOperationLogger("OP_03", "op log unit test", LogLevel::kDebug);
    EXPECT_TRUE(log3 != nullptr);
    log3->LogInfo() << "LogInfo 1.3";
    log3->LogDebug() << "LogDebug 1.3";
    log3->LogError() << "LogError 1.3";
}

// 输出第二批OP日志，包括 ctx (OP_02 OP_03 OP_04)
TEST_F(LogUnitTest009, LogOpTest02) {
    InitLogging("LOG_TEST_009_2", "log unit test 009", LogLevel::kDebug, HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, 10);

    auto log2 = CreateOperationLogger("OP_02", "op log unit test", LogLevel::kDebug);
    EXPECT_TRUE(log2 != nullptr);
    log2->LogInfo() << "LogInfo 2.2";
    log2->LogDebug() << "LogDebug 2.2";
    log2->LogError() << "LogError 2.2";

    auto log3 = CreateOperationLogger("OP_03", "op log unit test", LogLevel::kDebug);
    EXPECT_TRUE(log3 != nullptr);
    log3->LogInfo() << "LogInfo 2.3";
    log3->LogDebug() << "LogDebug 2.3";
    log3->LogError() << "LogError 2.3";

    auto log4 = CreateOperationLogger("OP_04", "op log unit test", LogLevel::kDebug);
    EXPECT_TRUE(log4 != nullptr);
    log4->LogInfo() << "LogInfo 2.4";
    log4->LogDebug() << "LogDebug 2.4";
    log4->LogError() << "LogError 2.4";
}

// OP日志首先会输出到普通日志中，其次会根据CTXID分成不同文件
// 检查普通的第一个日志文件
TEST_F(LogUnitTest009, LogOpTest03) {

    std::string directoryPath = "/opt/usr/log/soc_log/";
    std::string prefix = "LOG_TEST_009_1";
    std::string suffix = ".log";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string searchString1 = "LogError 1.1";
    std::string searchString2 = "LogError 1.2";
    std::string searchString3 = "LogError 1.3";

    std::string searchString4 = "LogError 2.2";
    std::string searchString5 = "LogError 2.3";
    std::string searchString6 = "LogError 2.4";

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
                std::string fileContent = readFileToString(filePath);
                std::cout << "file str: " << fileContent << std::endl;

                EXPECT_TRUE(fileContent.find(searchString1) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString2) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString3) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString4) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString5) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString6) == std::string::npos);
                break;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
}
// 检查普通的第二个日志文件
TEST_F(LogUnitTest009, LogOpTest04) {

    std::string directoryPath = "/opt/usr/log/soc_log/";
    std::string prefix = "LOG_TEST_009_2";
    std::string suffix = ".log";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string searchString1 = "LogError 1.1";
    std::string searchString2 = "LogError 1.2";
    std::string searchString3 = "LogError 1.3";

    std::string searchString4 = "LogError 2.2";
    std::string searchString5 = "LogError 2.3";
    std::string searchString6 = "LogError 2.4";

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
                std::string fileContent = readFileToString(filePath);
                std::cout << "file str: " << fileContent << std::endl;

                EXPECT_TRUE(fileContent.find(searchString1) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString2) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString3) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString4) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString5) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString6) != std::string::npos);
                break;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
}
// 检查OP日志的第一个文件
TEST_F(LogUnitTest009, LogOpTest05) {

    std::string directoryPath = "/opt/usr/log/soc_log/";
    std::string prefix = "OP_01";
    std::string suffix = ".log";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string searchString1 = "LogInfo 1.1";
    std::string searchString2 = "LogDebug 1.1";
    std::string searchString3 = "LogError 1.1";

    std::string searchString4 = "LogError 2.2";
    std::string searchString5 = "LogError 2.3";
    std::string searchString6 = "LogError 2.4";

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
                std::string fileContent = readFileToString(filePath);
                std::cout << "file str: " << fileContent << std::endl;

                EXPECT_TRUE(fileContent.find(searchString1) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString2) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString3) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString4) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString5) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString6) == std::string::npos);
                break;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory." << std::endl;
    }
}

// 检查OP日志的第三个文件
TEST_F(LogUnitTest009, LogOpTest06) {

    std::string directoryPath = "/opt/usr/log/soc_log/";
    std::string prefix = "OP_03";
    std::string suffix = ".log";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string searchString1 = "LogInfo 1.3";
    std::string searchString2 = "LogInfo 2.3";
    std::string searchString3 = "LogError 2.3";

    std::string searchString4 = "LogDebug 1.2";
    std::string searchString5 = "LogError 2.2";
    std::string searchString6 = "LogError 2.4";

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
                std::string fileContent = readFileToString(filePath);
                std::cout << "file str: " << fileContent << std::endl;

                EXPECT_TRUE(fileContent.find(searchString1) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString2) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString3) != std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString4) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString5) == std::string::npos);
                EXPECT_TRUE(fileContent.find(searchString6) == std::string::npos);
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