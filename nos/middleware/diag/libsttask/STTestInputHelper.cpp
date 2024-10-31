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
 * @file STTestInputHelper.cpp
 * @brief implements of STTestInputHelper
 */


#include "STTestInputHelper.h"
#include <iostream>
#include <string.h>

namespace hozon {
namespace netaos {
namespace sttask {

    STTestInputHelper::STTestInputHelper()
    {
    }

    STTestInputHelper::~STTestInputHelper()
    {
    }

    void STTestInputHelper::addCommand(const char* name, STTestInputHelper::COMMAND_CB cb)
    {
        if (nullptr == name) {
            return;
        }
        if (nullptr == cb) {
            return;
        }
        COMMAND_INFO info;
        info.name = name;
        info.cb = cb;
        m_allCommands.push_back(info);
    }

    int STTestInputHelper::enterLoop()
    {
        while (1) {
            std::cout << "=================================" << std::endl;
            std::cout << " Please input command:           " << std::endl;
            std::cout << "=================================" << std::endl;

            std::string input;
            std::getline(std::cin, input);
            std::string cmd;
            int argc;
            std::string argv[MAX_ARG_COUNT];
            parseInput(input.c_str(), cmd, argc, argv);

            if (cmd == "exit") {
                break;
            }
            else if (cmd == "help") {
                help();
            }
            else {
                STTestInputHelper::COMMAND_CB cb = findCommandCB(cmd);
                if (cb) {
                    (*this.*cb)(argc, argv);
                    continue;
                }
                else {
                    std::cout << "unknown command(" << cmd << ")." << std::endl;
                }
            }
        }

        return 0;
    }

    void STTestInputHelper::parseInput(const char* input, std::string& cmd, int& argc, std::string* argv)
    {
        static const char char_BLANK = ' ';
        static const char char_QUOTE = '"';

        int length = strlen(input);
        int pos = 0;
        argc = 0;
        while (pos < length && argc < MAX_ARG_COUNT) {
            if (char_BLANK == input[pos]) {
                pos++;
                continue;
            }

            int start = pos;
            bool inQuote = (char_QUOTE == input[pos++]) ? true : false;
            if (inQuote) {
                start++;
            }

            while (pos < length && (input[pos] != (inQuote ? char_QUOTE : char_BLANK))) {
                pos++;
            }

            int end = pos++;

            std::string arg(input, start, end - start);

            if (0 < argc) {
                argv[argc - 1] = arg;
            }
            else {
                cmd = arg;
            }

            argc++;
        }

        if (0 < argc) {
            argc--;
        }
        else {
            cmd = input;
        }
    }

    void STTestInputHelper::help()
    {
        int indexHelp = 0;
        std::cout << indexHelp++ << ". exit" << std::endl;
        std::cout << indexHelp++ << ". help" << std::endl;
        for (COMMAND_LIST::iterator it = m_allCommands.begin(); it != m_allCommands.end(); ++it) {
            const COMMAND_INFO& info = *it;
            std::cout << indexHelp++ << ". " << info.name << std::endl;
        }
    }

    STTestInputHelper::COMMAND_CB  STTestInputHelper::findCommandCB(const std::string& name)
    {
        for (COMMAND_LIST::iterator it = m_allCommands.begin(); it != m_allCommands.end(); ++it) {
            const COMMAND_INFO& info = *it;
            if (name == info.name) {
                return info.cb;
            }
        }
        return nullptr;
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */