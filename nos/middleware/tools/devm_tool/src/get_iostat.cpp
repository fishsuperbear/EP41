

#include <iostream>
#include <vector>
#include "get_iostat.h"

namespace hozon {
namespace netaos {
namespace tools {

void
IostatInfo::PrintUsage()
{
    std::cout << "Usage: nos devm iostat" << std::endl;
    std::cout << "  used to monitor system input/output (I/O) performance." << std::endl;
    std::cout << std::endl;
}

// static int system_cmd(char *command)
// {
//     char buffer[128];

//     FILE *fp = popen(command, "r");
//     if (fp == NULL) {
//         perror("popen");
//         return 1;
//     }

//     // 从命令输出中读取内容并存储在 buffer 中
//     while (fgets(buffer, sizeof(buffer), fp) != NULL) {
//         printf("%s", buffer);  // 可以根据需要处理输出，比如存储到其他变量中
//     }

//     pclose(fp);

//     return 0;
// }

int32_t
IostatInfo::StartGetIostat()
{
    const char *command = "iostat -p";
    char buffer[128];

    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        perror("popen");
        return -1;
    }

    // 从命令输出中读取内容并存储在 buffer 中
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        printf("%s", buffer);  // 可以根据需要处理输出，比如存储到其他变量中
    }

    pclose(fp);

    return 0;
}


}
}
}

