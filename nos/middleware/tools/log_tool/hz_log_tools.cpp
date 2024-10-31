#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <thread>
#include <sys/stat.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <unzipper.h>

const std::string OPT_SETLOGLEVEL = "setLogLevel";
const std::string OPT_CONTINUE = "continue";
const std::string OPT_CONVERT = "convert";

using namespace zipper;

void LogSetBySocketThread(std::string msg)
{
    // 1.创建一个通信的socket
    int fd = socket(AF_INET, SOCK_DGRAM, 0);

    if (fd == -1)
    {
        std::cout << "socket() error" << std::endl;
        return;
    }

    //2.客户端绑定本地的IP和端口号
    struct sockaddr_in remote_addr;
    memset(&remote_addr, 0, sizeof(remote_addr));
    remote_addr.sin_family = AF_INET;
    remote_addr.sin_port = htons(58297);
    remote_addr.sin_addr.s_addr = inet_addr("224.0.0.55");


    sendto(fd,  msg.c_str(), msg.size(), 0, (struct sockaddr *)&remote_addr, sizeof(remote_addr));


    close(fd);
}


// 打印 help
void printHelp()
{
    std::cout << R"(
        用法: 
                nos log [参数] [log文件路径]      持续输出指定的日志文件
        或者     nos log [参数] [日志Level信息]    修改日志等级
        或者     nos log [参数] [被压缩的日志]     内存读取被压缩的日志

        参数:
        setLogLevel			设定等级
        continue			持续输出日志 

        举例：
        1. nos log setLogLevel HZ_TEST.IGNORE:kError
        2. nos log continue /opt/usr/log/HZ_TEST_0000_2023-04-01_04-56-15.log
        3. nos log convert /log/DIAG_0000_2023-09-18_14-35-59.zip
    )" << std::endl;
}

bool compareIgnoreCase(const std::string& str1, const std::string& str2) {
    if (str1.length() != str2.length()) {
        return false;
    }
    return std::equal(str1.begin(), str1.end(), str2.begin(),
                      [](char c1, char c2) { return std::toupper(c1) == std::toupper(c2); });
}

bool FileExists(const std::string& fileName) {
    std::ifstream infile(fileName);
    return infile.good();
}

bool continueOutput(const std::string& log_path)
{
    if(FileExists(log_path)) {
        // go on 
    } else {
        std::cout << log_path << " does not exist.\n";
        return false;
    }
    std::string prefix_cmd = "tail -f ";
    std::string cmd = prefix_cmd + log_path;
    auto res = system(cmd.c_str());
    std::cout << "res is: " << res << std::endl;
    return true;
}

bool isZipFile(const std::string& filename) {
    size_t dotPos = filename.find_last_of(".");
    if (dotPos != std::string::npos) {
        std::string extension = filename.substr(dotPos + 1);
        if (extension == "zip") {
            return true;
        }
    }
    return false;
}

void convertLog(const std::string zipPath)
{
    if (!isZipFile(zipPath))
    {
        std::string cmdStr = "less " + zipPath;
        if (0 == system(cmdStr.c_str())){
        }
        return;
    }
    
    Unzipper unzipper(zipPath);
    std::vector<unsigned char> unzipped_entry;
    std::vector<ZipEntry> entries = unzipper.entries();
    if (entries.empty() || entries.size() > 1) {
        std::cout << "convert error, please check zip format !" << std::endl;
        unzipper.close();
        return;
    }
    for (auto && entry : entries)
    {
        if (entry.valid())
        {
            // std::cout << "log.name : " << entry.name << std::endl;
            // std::cout << "log.timestamp : " << entry.timestamp << std::endl;
            // std::cout << "log.compressedSize : " << entry.compressedSize << std::endl;
            // std::cout << "log.uncompressedSize : " << entry.uncompressedSize << std::endl;
            // std::cout << "log.dosdate : " << entry.dosdate << std::endl;
            // std::cout << "log.valid : " << entry.valid() << std::endl;
        } else {
            std::cout << "invalid !!" << std::endl;
        }
    }
    auto res = unzipper.extractEntryToMemory(entries.at(0).name, unzipped_entry);
    if (!res)
    {
        std::cerr << "convert error, zip read error !" << std::endl;
        unzipper.close();
        return;
    }
    unzipper.close();
    
    std::string longText(unzipped_entry.begin(), unzipped_entry.end());

    std::ofstream tempFile("/tmp/convert_logfile.txt");
    if (tempFile.is_open()) {
        tempFile << longText;
        tempFile.close();
        
        if (0 == system("less /tmp/convert_logfile.txt")){
        }

        std::remove("/tmp/convert_logfile.txt");
    } else {
        std::cerr << "error, can not create tmp file." << std::endl;
    }
}

int main(int argc, char * argv[])
{

    if (argc != 3)
    {
        std::cout << "Please check argv numbers, refer to the following help: " << std::endl;
        printHelp();
        return -1;
    }

    std::string operate = argv[1];
    if (compareIgnoreCase(operate, OPT_SETLOGLEVEL))
    {
        std::string msg = argv[2];
        std::thread changeLevel(LogSetBySocketThread,  msg);
        changeLevel.join();
    } 
    else if (compareIgnoreCase(operate, OPT_CONTINUE))
    {
        std::string msg = argv[2];
        if(!continueOutput(msg))
        {
            return -1;
        }
    }
    else if (compareIgnoreCase(operate, OPT_CONVERT))
    {
        std::string msg = argv[2];
        convertLog(msg);
    }
    else 
    {
        std::cout << "Please check operate msg, refer to the following help: " << std::endl;
        printHelp();
        return -1;
    }

    return 0;
}
