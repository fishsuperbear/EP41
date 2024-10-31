// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CCmdLineParser.hpp"

SIPLStatus CCmdLineParser::Parse(int argc, char *argv[], CAppConfig *pAppConfig)
{
#if !NV_IS_SAFETY
    const char *const short_options = "hg:m:v:t:lN:deIpPfr:c:C:k:q:V";
#else
    const char *const short_options = "hv:t:lN:deIpPfr:c:C:k:q:V";
#endif
    const struct option long_options[] = {
        // clang-format off
        { "help",                 no_argument,       0, 'h' },
#if !NV_IS_SAFETY
        { "platform-config",      required_argument, 0, 'g' },
        { "link-enable-masks",    required_argument, 0, 'm' },
        { "late-attach",          no_argument,       0, 'L' },
#endif
        { "verbosity",            required_argument, 0, 'v' },
        { "nito",                 required_argument, 0, 'N' },
        { "filedump",             no_argument,       0, 'f' },
        { "frameFilter",          required_argument, 0, 'k' },
        { "version",              no_argument,       0, 'V' },
        { "runfor",               required_argument, 0, 'r' },
        { "enableDisplay",        no_argument,       0, 'd' },
        { "multiElem",            no_argument,       0, 'e' },
        { 0,                      0,                 0,  0  }
        // clang-format on
    };

    int index = 0;
    bool bShowHelp = false;
    bool bShowConfigs = false;
    string sConsumerType = "";
    string sQueueType = "";

    while (1) {
        const auto getopt_ret = getopt_long(argc, argv, short_options, &long_options[0], &index);
        if (getopt_ret == -1) {
            // Done parsing all arguments.
            break;
        }

        switch (getopt_ret) {
            default:  /* Unrecognized option */
            case '?': /* Unrecognized option */
                cout << "Invalid or Unrecognized command line option. Specify -h or --help for options\n";
                bShowHelp = true;
                break;
            case 'h': /* -h or --help */
                bShowHelp = true;
                break;
#if !NV_IS_SAFETY
            case 'g':
                pAppConfig->m_sDynamicConfigName = optarg;
                break;
            case 'm': {
                char *token = std::strtok(optarg, " ");
                while (token != NULL) {
                    pAppConfig->m_vMasks.push_back(stoi(token, nullptr, 16));
                    token = std::strtok(NULL, " ");
                }
            } break;
            case 'L':
                pAppConfig->m_bEnableLateAttach = true;
                break;
#endif
            case 'v':
                pAppConfig->m_uVerbosity = atoi(optarg);
                break;
            case 't':
                pAppConfig->m_sStaticConfigName = optarg;
                break;
            case 'l':
                bShowConfigs = true;
                break;
            case 'N':
                pAppConfig->m_sNitoFolderPath = optarg;
                break;
            case 'I':
                pAppConfig->m_bIgnoreError = true;
                break;
            case 'p': /* set producer resident */
                pAppConfig->m_eCommType = CommType_InterProcess;
                pAppConfig->m_eEntityType = EntityType_Producer;
                break;
            case 'c': /* set consumer resident */
                pAppConfig->m_eCommType = CommType_InterProcess;
                pAppConfig->m_eEntityType = EntityType_Consumer;
                sConsumerType = optarg;
                break;
            case 'P': /* set C2C producer resident */
                pAppConfig->m_eCommType = CommType_InterChip;
                pAppConfig->m_eEntityType = EntityType_Producer;
                break;
            case 'C': /* set C2C consumer resident */
                pAppConfig->m_eCommType = CommType_InterChip;
                pAppConfig->m_eEntityType = EntityType_Consumer;
                sConsumerType = optarg;
                break;
            case 'f':
                pAppConfig->m_bFileDump = true;
                break;
            case 'k':
                pAppConfig->m_uFrameFilter = atoi(optarg);
                break;
            case 'r':
                pAppConfig->m_uRunDurationSec = atoi(optarg);
                break;
            case 'q':
                sQueueType = optarg;
                break;
            case 'V':
                pAppConfig->m_bShowVersion = true;
                break;
            case 'd':
                pAppConfig->m_bEnableDisplay = true;
                break;
            case 'e':
                pAppConfig->m_bEnableMultiElements = true;
                break;
        }
    }

    if (bShowHelp) {
        ShowUsage();
        return NVSIPL_STATUS_EOF;
    }

    if (bShowConfigs) {
        // User just wants to list available configs
        ShowConfigs();
        return NVSIPL_STATUS_EOF;
    }

    // Display is currently not supported for NvSciBufPath
    if ((pAppConfig->m_uFrameFilter < 1) || (pAppConfig->m_uFrameFilter > 5)) {
        cout << "Invalid value of frame filter, the range is 1-5\n";
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (sConsumerType != "") {
        auto status = ParseConsumerTypeStr(sConsumerType, pAppConfig->m_eConsumerType);
        CHK_STATUS_AND_RETURN(status, "ParseConsumerTypeStr");
    }

    if (sQueueType != "") {
        auto status = ParseQueueTypeStr(sQueueType, pAppConfig->m_eQueueType);
        CHK_STATUS_AND_RETURN(status, "ParseQueueTypeStr");
    }

#if !NV_IS_SAFETY
    if ((pAppConfig->m_sDynamicConfigName != "" && pAppConfig->m_vMasks.size() == 0) ||
        (pAppConfig->m_sDynamicConfigName == "" && pAppConfig->m_vMasks.size() > 0)) {
        cout << "Dynamic config and link masks must be set together\n";
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if ((pAppConfig->m_sDynamicConfigName != "") && (pAppConfig->m_sStaticConfigName != "")) {
        cout << "Dynamic config and static config couldn't be set together\n";
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
#endif

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCmdLineParser::ParseConsumerTypeStr(string sConsumerType, ConsumerType &eConsumerType)
{
    if (sConsumerType == "cuda") {
        eConsumerType = ConsumerType_Cuda;
    } else if (sConsumerType == "enc") {
        eConsumerType = ConsumerType_Enc;
    } else {
        cout << "Invalid consumer type! supported: 'enc': encoder consumer, 'cuda': cuda consumer\n";
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CCmdLineParser::ParseQueueTypeStr(string sQueueType, QueueType &eQueueType)
{
    if ((sQueueType == "m") || (sQueueType == "M")) {
        eQueueType = QueueType_Mailbox;
    } else if ((sQueueType == "f") || (sQueueType == "F")) {
        eQueueType = QueueType_Fifo;
    } else {
        cout << "Invalid value of queue tpye! range:[f|F|m|M]\n";
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return NVSIPL_STATUS_OK;
}

void CCmdLineParser::ShowUsage(void)
{
    // clang-format off
    cout << "Usage:\n";
    cout << "-h or --help                               :Prints this help\n";
#if !NV_IS_SAFETY
    cout << "-g or --platform-config 'name'             :Specify dynamic platform configuration, which is fetched via SIPL Query.\n";
    cout << "--link-enable-masks 'masks'                :Enable masks for links on each deserializer connected to CSI\n";
    cout << "                                           :masks is a list of masks for each deserializer.\n";
    cout << "                                           :Eg: '0x0000 0x1101 0x0000 0x0000' disables all but links 0, 2 and 3 on CSI-CD intrface\n";
    cout << "-L or --late-attach                        :Enable late-attach/re-attach\n";
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
    cout << "-t <platformCfgName>                       :Specify static platform configuration, which is defined in header files, default is F008A120RM0AV2_CPHY_x4\n";
    cout << "-l                                         :List supported configs \n";
    cout << "--nito <folder>                            :Path to folder containing NITO files\n";
    cout << "-I                                         :Ignore the fatal error\n";
    cout << "-p                                         :Inter-process: Producer resides in this process\n";
    cout << "-c 'type'                                  :Inter-process: Consumer resides in this process.\n";
    cout << "                                           :Supported type: 'enc': encoder consumer, 'cuda': cuda consumer.\n";
    cout << "-P                                         :C2C: Producer resides in this process\n";
    cout << "-C 'type'                                  :C2C: Consumer resides in this process.\n";
    cout << "                                           :Supported type: 'enc': encoder consumer, 'cuda': cuda consumer.\n";
    cout << "-f or --filedump                           :Dump output to file on each consumer side\n";
    cout << "-k or --frameFilter <n>                    :Process every Nth frame, range:[1, 5], (default: 1)\n";
    cout << "-q 'f|F|m|M'                               :use fifo (f|F) or maibox (m|M) for consumer [default f]\n";
    cout << "-V or --version                            :Show version\n";
    cout << "-r or --runfor <seconds>                   :Exit application after n seconds\n";
    cout << "-d                                         :enable stitching cameras and display\n";
    cout << "-e or --multiElem                          :enable ISP0&ISP1 output to activate multiple elements usecase\n";
    cout << "                                           :ISP0 output --> encoder/display consumer, ISP1 output ---> cuda consumer\n";
    // clang-format on
    return;
}

void CCmdLineParser::ShowConfigs(void)
{
#if !NV_IS_SAFETY
    cout << "Dynamic platform configurations:\n";
    auto pQuery = INvSIPLQuery::GetInstance();
    if (pQuery == nullptr) {
        cout << "INvSIPLQuery::GetInstance() failed\n";
    }

    auto status = pQuery->ParseDatabase();
    if (status != NVSIPL_STATUS_OK) {
        cout << "INvSIPLQuery::ParseDatabase failed\n";
    }

    for (auto &cfg : pQuery->GetPlatformCfgList()) {
        cout << "\t" << std::setw(35) << std::left << cfg->platformConfig << ":" << cfg->description << endl;
    }
    cout << "Static platform configurations:\n";
#endif
    cout << "\t" << std::setw(35) << std::left << platformCfgAr0820.platformConfig << ":"
         << platformCfgAr0820.description << endl;
}
