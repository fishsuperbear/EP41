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
 * @file  STLog.h
 * @brief Class of STLog
 */
#ifndef STLOG_H
#define STLOG_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <stdarg.h>
#include <stdint.h>

namespace hozon {
namespace netaos {
namespace sttask {


    /**
     * @brief Class of STLog
     *
     * This class will accept the command from proxy.
     */
    class STLog
    {
    public:
        STLog() {}
        virtual ~STLog() {}
        static void output(const char* tag,
                            const char *subtag,
                            uint8_t type,
                            const char* func,
                            const char* file,
                            uint32_t line,
                            const char* format,
                            ...);

    private:
        STLog(const STLog&);
        STLog& operator=(const STLog&);
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STLOG_H */
/* EOF */