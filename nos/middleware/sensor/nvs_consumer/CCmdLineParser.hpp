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

#include "NvSIPLTrace.hpp" // NvSIPLTrace to set library trace level

#ifndef CCMDLINEPARSER_HPP
#define CCMDLINEPARSER_HPP

using namespace std;
using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace desay { 

class CCmdLineParser
{
 public:
    // Command line options
    uint32_t verbosity = 1u;
    string sPlatformCfgName = "DSV_multiple_camera";
    uint32_t mutlticastNum = 1;
    bool bShowFPS = false; 
    string sConsumerType = "";
    uint32_t mutlticastIndex = 0;
    
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
        cout << "-c 'type'                                  :consumer resides in this process.\n";
        cout << "                                           :Supported type: 'enc': encoder consumer, 'cuda': cuda consumer,if no set,use media consumer\n";
        cout << "-m                                         :multicust count,need as same as nvsipl multicaset app setting\n";
        cout << "-s                                         :Show FPS (frames per second) every 2 seconds\n";
        cout << "-i                                         :this consumer use multicast index\n";
        return;
    }

    int Parse(int argc, char* argv[])
    {
        const char* const short_options = "hv:t:c:m:si:";
        const struct option long_options[] =
        {
            { "help",                 no_argument,       0, 'h' },
            { "verbosity",            required_argument, 0, 'v' },
            { "nito",                 required_argument, 0, 'N' },
            { "mulCount",             required_argument, 0, 'm' },
            { "showfps",              no_argument,       0, 's' },
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
            case 't':
                sPlatformCfgName = string(optarg);
                break;
            case 'c':
                sConsumerType = string(optarg);
                break;
            case 'm':
                mutlticastNum = atoi(optarg);
                if(mutlticastNum > 8){
                    mutlticastNum = 8;
                    cout << "mutlticastNum max is 8\n";
                }
                break;
            case 'i':
                mutlticastIndex = atoi(optarg);
                break;
            case 's':
                bShowFPS = true; 
            }
        }

        if (bShowHelp) {
            ShowUsage();
            return -1;
        }
        return 0;
    }
};

}
}
}

#endif //CCMDPARSER_HPP
