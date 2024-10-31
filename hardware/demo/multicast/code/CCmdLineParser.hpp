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
#if !NV_IS_SAFETY
#include "NvSIPLQuery.hpp" // NvSIPLQuery to display platform config
#endif
#include "platform/ar0820.hpp"

#ifndef CCMDLINEPARSER_HPP
#define CCMDLINEPARSER_HPP

using namespace std;
using namespace nvsipl;

class CCmdLineParser
{
 public:
    // Command line options
    uint32_t verbosity = 1u;

#if !NV_IS_SAFETY
    string sDynamicConfigName = "";
    vector<uint32_t> vMasks;
#endif
    string sStaticConfigName = "";

#ifdef NVMEDIA_QNX
    string sNitoFolderPath = "/proc/boot/";
#else
    string sNitoFolderPath = "/opt/nvidia/nvmedia/nit/";
#endif

    bool bMultiProcess = false;
    bool bIsProducer = false;
    bool bIsConsumer = false;
    string sConsumerType = "";
    bool bIgnoreError = false;
    bool bFileDump = false;
    uint8_t uFrameMode = 1;
    bool bUseMailbox = false;
    string sQueueMode = "f";
    bool bShowVersion = false;
    uint32_t uRunDurationSec = 0;

    static void ShowUsage(void)
    {
        cout << "Usage:\n";
        cout << "-h or --help                               :Prints this help\n";
#if !NV_IS_SAFETY
        cout << "-g or --platform-config 'name'             :Specify dynamic platform configuration, which is fetched via SIPL Query.\n";
        cout << "--link-enable-masks 'masks'                :Enable masks for links on each deserializer connected to CSI\n";
        cout << "                                           :masks is a list of masks for each deserializer.\n";
        cout << "                                           :Eg: '0x0000 0x1101 0x0000 0x0000' disables all but links 0, 2 and 3 on CSI-CD intrface\n";
#endif // !NV_IS_SAFETY
        cout << "-v or --verbosity <level>                  :Set verbosity\n";
#if !NV_IS_SAFETY
        cout << "                                           :Supported values (default: 1)\n";
        cout << "                                           : " << INvSIPLTrace::LevelNone << " (None)\n";
        cout << "                                           : " << INvSIPLTrace::LevelError << " (Errors)\n";
        cout << "                                           : " << INvSIPLTrace::LevelWarning << " (Warnings and above)\n";
        cout << "                                           : " << INvSIPLTrace::LevelInfo << " (Infos and above)\n";
        cout << "                                           : " << INvSIPLTrace::LevelDebug << " (Debug and above)\n";
#endif // !NV_IS_SAFETY
        cout << "-t <platformCfgName>                       :Specify static platform configuration, which is defined in header files, default is F008A120RM0A_CPHY_x4\n";
        cout << "-l                                         :List supported configs \n";
        cout << "--nito <folder>                            :Path to folder containing NITO files\n";
        cout << "-I                                         :Ignore the fatal error\n";
        cout << "-p                                         :Producer resides in this process\n";
        cout << "-c 'type'                                  :Consumer resides in this process.\n";
        cout << "                                           :Supported type: 'enc': encoder consumer, 'cuda': cuda consumer.\n";
        cout << "-f or --filedump                           :Dump output to file on each consumer side\n";
        cout << "-k or --frameMode <n>                      :Process every Nth frame, range:[1, 5], (default: 1)\n";
        cout << "-q 'f|F|m|M'                               :use fifo (f|F) or maibox (m|M) for consumer [default f]\n";
        cout << "-V or --version                            :Show version\n";
        cout << "-r or --runfor <seconds>                   :Exit application after n seconds\n";
        return;
    }

    static void ShowConfigs(void)
    {
#if !NV_IS_SAFETY
        cout << "Dynamic platform configurations:\n";
        auto pQuery = INvSIPLQuery::GetInstance();
        if (pQuery == nullptr) {
            cout << "INvSIPLQuery::GetInstance() failed\n";
        }

        auto status = pQuery->ParseDatabase();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLQuery::ParseDatabase failed\n");
        }

        for (auto &cfg : pQuery->GetPlatformCfgList()) {
            cout << "\t" << std::setw(35) << std::left << cfg->platformConfig << ":" << cfg->description << endl;
        }
        cout << "Static platform configurations:\n";
#endif
        cout << "\t" << std::setw(35) << std::left << platformCfgAr0820.platformConfig << ":" << platformCfgAr0820.description << endl;
    }

    int Parse(int argc, char* argv[])
    {
#if !NV_IS_SAFETY
        const char* const short_options = "hg:m:v:t:lN:Ipfr:c:k:q:V";
#else
        const char* const short_options = "hv:t:lN:Ipfr:c:k:q:V";
#endif
        const struct option long_options[] =
        {
            { "help",                 no_argument,       0, 'h' },
#if !NV_IS_SAFETY
            { "platform-config",      required_argument, 0, 'g' },
            { "link-enable-masks",    required_argument, 0, 'm' },
#endif
            { "verbosity",            required_argument, 0, 'v' },
            { "nito",                 required_argument, 0, 'N' },
            { "filedump",             no_argument,       0, 'f' },
            { "frameMode",            required_argument, 0, 'k' },
            { "version",              no_argument,       0, 'V' },
            { "runfor",               required_argument, 0, 'r' },
            { 0,                      0,                 0,  0 }
        };

        int index = 0;
        auto bShowHelp = false;
        auto bShowConfigs = false;

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
#if !NV_IS_SAFETY
            case 'g':
                sDynamicConfigName = string(optarg);
                break;
            case 'm':
            {
                char* token = std::strtok(optarg, " ");
                while(token != NULL) {
                    vMasks.push_back(stoi(token, nullptr, 16));
                    token = std::strtok(NULL, " ");
                }
            }
                break;
#endif
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 't':
                sStaticConfigName = string(optarg);
                break;
            case 'l':
                bShowConfigs = true;
                break;
            case 'N':
                sNitoFolderPath = string(optarg);
                break;
            case 'I':
                bIgnoreError = true;
                break;
            case 'p': /* set producer resident */
                bIsProducer = true;
                bMultiProcess = true;
                break;
            case 'c': /* set consumer resident */
                bMultiProcess = true;
                bIsConsumer = true;
                sConsumerType = string(optarg);
                break;
            case 'f':
                bFileDump = true;
                break;
            case 'k':
                uFrameMode = atoi(optarg);
                break;
            case 'r':
                uRunDurationSec = atoi(optarg);
                break;
            case 'q':
                sQueueMode = string(optarg);
                break;
            case 'V':
                bShowVersion = true;
                break;
            }
        }

        if (bShowHelp) {
            ShowUsage();
            return -1;
        }

        if (bShowConfigs) {
            // User just wants to list available configs
            ShowConfigs();
            return -1;
        }

        // Display is currently not supported for NvSciBufPath
        if ((uFrameMode < 1) || (uFrameMode > 5)) {
            cout << "Invalid value of frame mode, the range is 1-5\n";
            return -1;
        }

        if ((sQueueMode == "m") || (sQueueMode == "M")) {
            bUseMailbox = true;
        } else if ((sQueueMode == "f") || (sQueueMode == "F")) {
            bUseMailbox = false;
        } else {
            cout << "Invalid value of queue mode! range:[f|F|m|M]\n";
            return -1;
        }
#if !NV_IS_SAFETY
        if ((sDynamicConfigName != "") && (sStaticConfigName != "")) {
            cout << "Dynamic config and static config couldn't be set together.\n";
            return -1;
        }
#endif

        return 0;
    }
};

#endif //CCMDPARSER_HPP
