/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class S19_TO_BIN Header
 */

#ifndef __S19_TO_BIN_H__
#define __S19_TO_BIN_H__

#include <stdint.h>
#include <iostream>
#include <vector>

#include "update_manager/file_to_bin/to_bin_base.h"

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief class S19ToBin
     *
     * S19 file to bin file class definition.
     */
    class S19ToBin : public ToBinBase
    {
    public:
        S19ToBin(std::string s19_file_in, std::string bin_file_out);
        ~S19ToBin(){}

        virtual std::int32_t Transition();

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

        std::int32_t parseLine(const std::uint8_t & addrLength, const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);

        std::int32_t parseS0Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS1Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS2Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS3Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS4Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS5Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS6Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS7Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS8Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);
        std::int32_t parseS9Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum);


    private:
        std::uint32_t m_bin_file_size;
        std::uint32_t m_flash_start_addr;
        std::uint32_t m_flash_end_addr;
        std::string m_file_in;
        std::string m_file_out;
        std::vector<FlashData> m_data_to_flash;
    };


} // end of update
} // end of netaos
} // end of hozon
#endif  // __S19_TO_BIN_H__
/* EOF */
