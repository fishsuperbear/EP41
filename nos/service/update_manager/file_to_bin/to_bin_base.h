/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class TO_BIN_BASE Header
 */

#ifndef __TO_BIN_BASE_H__
#define __TO_BIN_BASE_H__

#include <stdint.h>
#include <iostream>
#include <vector>

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class ToBinBase
     *
     * Hex file to bin file class definition.
     */
    class ToBinBase
    {
    public:
        ToBinBase(){}
        virtual ~ToBinBase(){}

        virtual std::int32_t Transition() = 0;
        virtual std::string GetInputFileName() = 0;
        virtual std::string GetBinFileName() = 0;
        virtual std::uint32_t GetBinFileSize() = 0;
        virtual std::uint32_t GetFlashStartAddr() = 0;
        virtual std::uint32_t GetFlashEndAddr() = 0;

    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // __TO_BIN_BASE_H__
/* EOF */
