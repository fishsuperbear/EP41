/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * @file  STTestInputHelper.h
 * @brief Class of STTestInputHelper
 */
#ifndef STTESTINPUTHELPER_H
#define STTESTINPUTHELPER_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <string>
#include <map>
#include <vector>

namespace hozon {
namespace netaos {
namespace sttask {

    /**
     * @brief Class of STTestInputHelper
     *
     * TBD.
     */
    class STTestInputHelper
    {
    public:
        static const int MAX_ARG_COUNT = 5;
        typedef void (STTestInputHelper::*COMMAND_CB)(int argc, std::string* argv);

        STTestInputHelper();
        virtual ~STTestInputHelper();

        void        addCommand(const char* name, STTestInputHelper::COMMAND_CB cb);
        int         enterLoop();

    protected:
        void        parseInput(const char* input, std::string& cmd, int& argc, std::string* argv);
        void        help();
        STTestInputHelper::COMMAND_CB  findCommandCB(const std::string& name);

    private:
        struct COMMAND_INFO
        {
            std::string name;
            COMMAND_CB cb;
        };

        typedef std::vector<COMMAND_INFO> COMMAND_LIST;
        COMMAND_LIST                 m_allCommands;

    private:
        STTestInputHelper(const STTestInputHelper&);
        STTestInputHelper& operator=(const STTestInputHelper&);
    };



} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STTESTINPUTHELPER_H */
/* EOF */