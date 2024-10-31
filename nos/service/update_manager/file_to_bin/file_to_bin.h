/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class FILE_TO_BIN Header
 */

#ifndef __FILE_TO_BIN_H__
#define __FILE_TO_BIN_H__

#include <stdint.h>
#include <iostream>
#include <vector>
#include <memory>
#include "update_manager/file_to_bin/to_bin_base.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class FileToBin
     *
     * Hex file to bin file class definition.
     */

    class FileToBin
    {
    public:
        FileToBin();
        ~FileToBin(){}

        bool Transition(std::string file_in, std::string bin_file_path);
        std::string GetBinFileName(std::string file_in);
        std::uint32_t GetBinFileSize(std::string file_in);
        std::uint32_t GetFlashStartAddr(std::string file_in);
        std::uint32_t GetFlashEndAddr(std::string file_in);

    private:
        std::vector<std::unique_ptr<ToBinBase>> m_toBin;

    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // __FILE_TO_BIN_H__
/* EOF */
