/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <iomanip>
#include <vector>

#include "NvSIPLTrace.hpp" // NvSIPLTrace to set library trace level

#ifndef CCMDLINEPARSER_HPP
#define CCMDLINEPARSER_HPP

using namespace std;
using namespace nvsipl;

class CCmdLineParser
{
 public:
    // Command line options
    uint32_t verbosity = 1u;
    string sNitoFolderPath = "";
    string sPlatformCfgName = "DSV_multiple_camera";
    string PlatformCfgFile = "/app/runtime_service/nvs_producer/conf/nvs_producer.yaml";
    bool bIsConsumer = false;
    bool bIgnoreError = false;
    uint32_t mutlticastNum = 1;
    bool bShowFPS = false; 
    std::vector<uint32_t> vMasks;

    static void ShowUsage(void)
    {
        cout << "Usage:\n";
        cout << "-h or --help                               :Prints this help\n";
        cout << "-v or --verbosity <level>                  :Set verbosity\n";
#if !NV_IS_SAFETY
        cout << "                                           :Supported values (default: 1)\n";
        cout << "                                           : " << INvSIPLTrace::LevelNone << " (None)\n";
        cout << "                                           : " << INvSIPLTrace::LevelError << " (Errors)\n";
        cout << "                                           : " << INvSIPLTrace::LevelWarning << " (Warnings and above)\n";
        cout << "                                           : " << INvSIPLTrace::LevelInfo << " (Infos and above)\n";
        cout << "                                           : " << INvSIPLTrace::LevelDebug << " (Debug and above)\n";
#endif // !NV_IS_SAFETY
        cout << "-t <platformCfgName>                       :Specify platform configuration, default is DSV_multiple_camera\n";
        cout << "--nito <folder>                            :Path to folder containing NITO files\n";
        cout << "-I                                         :Ignore the fatal error\n";
        cout << "-m                                         :multicust count,when camera count is 7, max is 3\n";
        cout << "-s                                         :Show FPS (frames per second) every 2 seconds\n";
        cout << "-c  or --config                            :platformconfig yaml file\n";
        return;
    }

    int Parse(int argc, char* argv[])
    {
        const char* const short_options = "hv:t:l:N:Im:sc:";
        const struct option long_options[] =
        {
            { "help",                 no_argument,       0, 'h' },
            { "verbosity",            required_argument, 0, 'v' },
            { "link-enable-masks",    required_argument, 0, 'l' },
            { "nito",                 required_argument, 0, 'N' },
            { "mulCount",             required_argument, 0, 'm' },
            { "showfps",              no_argument,       0, 's' },
            { "config",               required_argument, 0, 'c' },
            { 0,                      0,                 0,  0 }
        };

        int index = 0;
        auto bShowHelp = false;

        while (1) {
            const auto getopt_ret = getopt_long(argc, argv, short_options , &long_options[0], &index);
            if (getopt_ret == -1) {
                // Done parsing all arguments.
                break;
            }

            switch (getopt_ret) {
            default: /* Unrecognized option */
            case '?': /* Unrecognized option */
                cout << "Invalid or Unrecognized command line option. Specify -h or --help for options\n";
                bShowHelp = true;
                break;
            case 'h': /* -h or --help */
                bShowHelp = true;
                break;
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 'l':
                {
                    char *token = std::strtok(optarg, " ");
                    while (token != NULL) {
                        vMasks.push_back(stoi(token, nullptr, 16));
                        token = std::strtok(NULL, " ");
                    }
                }
                break;
            case 't':
                sPlatformCfgName = string(optarg);
                break;
            case 'N':
                sNitoFolderPath = string(optarg);
                break;
            case 'I':
                bIgnoreError = true;
                break;
            case 'm':
                mutlticastNum = atoi(optarg);
                if(mutlticastNum > 8){
                    mutlticastNum = 8;
                    cout << "mutlticastNum max is 8\n";
                }
                break;
            case 's':
                bShowFPS = true; 
            case 'c':
                PlatformCfgFile = string(optarg);
            }
        }

        if (bShowHelp) {
            ShowUsage();
            return -1;
        }
        return 0;
    }
};

#endif //CCMDPARSER_HPP
