/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class HEX_TO_BIN Header
 */

#ifndef __HEX_TO_BIN_H__
#define __HEX_TO_BIN_H__

#include <stdint.h>
#include <iostream>
#include <vector>

#include "update_manager/file_to_bin/to_bin_base.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class HexToBin
     *
     * Hex file to bin file class definition.
     */
    class HexToBin : public ToBinBase
    {
    public:
        HexToBin(std::string hex_file_in, std::string bin_file_out);
        ~HexToBin(){}

        std::int32_t Transition();

        virtual std::string GetInputFileName()
        {
            return m_file_in;
        }

        virtual std::string GetBinFileName()
        {
            return m_file_out;
        }

        virtual std::uint32_t GetBinFileSize()
        {
            return m_bin_file_size;
        }

        virtual std::uint32_t GetFlashStartAddr()
        {
            return m_flash_start_addr;
        }

        virtual std::uint32_t GetFlashEndAddr()
        {
            return m_flash_end_addr;
        }
        

    private:
        struct FlashData
        {
            std::uint32_t startAddr;
            std::vector<std::uint8_t> data;
        };

        std::int32_t parseLine(const std::string & line, std::uint8_t & dataSize, std::uint16_t & addr, std::uint8_t & type, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);



    private:
        std::uint32_t m_bin_file_size;
        std::uint32_t m_flash_start_addr;
        std::uint32_t m_flash_end_addr;
        std::uint32_t m_base_phy_addr;
        bool m_extended_linear_addr_record;
        std::string m_file_in;
        std::string m_file_out;
        std::vector<FlashData> m_data_to_flash;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // __HEX_TO_BIN_H__
/* EOF */
