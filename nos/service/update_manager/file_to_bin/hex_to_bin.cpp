/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CanCommandTask implement
 */
#include <fstream>
#include <string>
#include <iostream>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "update_manager/file_to_bin/hex_to_bin.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/common/common_operation.h"

#include <iostream>

namespace hozon {
namespace netaos {
namespace update {


    HexToBin::HexToBin(std::string s19_file_in, std::string bin_file_out)
        :ToBinBase()
        ,m_bin_file_size(0)
        ,m_flash_start_addr(0xFFFFFFFF)
        ,m_flash_end_addr(0)
        ,m_base_phy_addr(0)
        ,m_extended_linear_addr_record(true)
        ,m_file_in(s19_file_in)
        ,m_file_out(bin_file_out)
    {
        m_data_to_flash.clear();


    }

    std::int32_t HexToBin::parseLine(const std::string & line, std::uint8_t & dataSize, std::uint16_t & addr, std::uint8_t & type, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        //UPDATE_LOG_D(" HexToBin::parseLine start!";
        std::uint8_t start_pos = 1;
        std::uint16_t check_sum_count = 0;
        dataSize = std::stoi(line.substr(start_pos, 2), 0, 16);
        addr = std::stoi(line.substr(start_pos + 2, 4), 0, 16);


        type = std::stoi(line.substr(start_pos + 6, 2), 0, 16);

        if (1 == type)
        {
            return 0;
        }



        if ((2 * dataSize) != line.length() - 1 -2 -4 -2 -2 -1)
        {
            UM_ERROR << "line data error!";
            UM_ERROR << "dataSize :" << static_cast<uint16_t>(dataSize) << ", line.length() - 1 -2 -4 -2 -2: " << line.length() - 1 -2 -4 -2 -2;
            return -1;
        }

        check_sum_count = dataSize + ((addr >> 8) & 0xFF) + (addr  & 0xFF) + type;

        std::string strData = line.substr(start_pos + 8, (2 * dataSize));
        check_sum = std::stoi(line.substr(line.length() - 1 - 2, 2), 0, 16);

        //UPDATE_LOG_D("dataSize:" << static_cast<uint16_t>(dataSize) << ", addr:" << addr << ", check_sum:" << static_cast<uint16_t>(check_sum) << ", strData:" << strData.c_str();

        data.clear();

        for (std::uint16_t i = 0; (2 * i) < strData.size(); i++)
        {
            std::uint8_t tmpData = std::stoi(strData.substr((2 * i), 2), 0, 16);
            data.push_back(tmpData);
            check_sum_count = (check_sum_count + tmpData) & 0xFF;
        }

        if ((std::uint8_t)(check_sum) != (std::uint8_t)(0xff & ( 0x100- check_sum_count)))
        {
            UM_ERROR << "HexToBin::parseLine check sum error, check_sum_count:" << check_sum_count << " , read check_sum:" << check_sum;
            return -2;
        }

        return 0;
    }

    std::int32_t HexToBin::Transition()
    {
        auto pos_hex = m_file_in.find(".hex");

        if (pos_hex == std::string::npos)
        {
            pos_hex = m_file_in.find(".HEX");
            if (pos_hex == std::string::npos)
            {
                UM_ERROR << "file : " << m_file_in.c_str() << "not a hex file!";
                return -1;
            }
        }

        std::ifstream inFile(m_file_in.c_str(), std::ios::in);
        if (!inFile)
        {
            UM_ERROR << "file : " << m_file_in.c_str() << "open error!";
            return -1;
        }



        std::string strline;
        std::string str_Icon;
        std::uint8_t type = 0;
        std::uint8_t dataSize = 0;
        std::uint16_t addr = 0;
        std::uint32_t addrPhy = 0xFFFFFFFF;
        std::uint8_t check_sum = 0;
        std::int32_t ret = -1;

        while(std::getline(inFile, strline))
        {
            std::vector<std::uint8_t> data;

            if (strline.length() < 11)
            {
                UM_ERROR << "the length of line error!";
                return -2;
            }

            //UPDATE_LOG_D("strline: " << strline.c_str();

            str_Icon = strline.substr(0, 1);



            if (str_Icon != std::string(":"))
            {
                UM_ERROR << "Not start with : !";
                return -3;
            }

            data.clear();

            ret = parseLine(strline, dataSize, addr, type, data, check_sum);
            if (0 != ret)
            {
                UM_ERROR << "parseLine error!";
                return -4;
            }


            switch (type)
            {
                case 0:
                    //UPDATE_LOG_D("m_base_phy_addr: 0x" << hozon::netaos::log::LogHex32{m_base_phy_addr};
                    if (m_extended_linear_addr_record)
                    {
                        addrPhy = m_base_phy_addr + (addr & 0xFFFF);
                    }
                    else
                    {
                        addrPhy = m_base_phy_addr + (addr & 0x000F);
                    }

                    //UPDATE_LOG_D("addrPhy: 0x" << hozon::netaos::log::LogHex16{addrPhy};


                    m_flash_start_addr = m_flash_start_addr < addrPhy ? m_flash_start_addr : addrPhy;
                    m_flash_end_addr = m_flash_end_addr > (addrPhy + data.size() -1) ?  m_flash_end_addr : (addrPhy + data.size()  -1);

                    //"m_flash_start_addr: 0x" << hozon::netaos::log::LogHex32{m_flash_start_addr};
                    //UPDATE_LOG_D("m_flash_end_addr: 0x" << hozon::netaos::log::LogHex32{m_flash_end_addr};

                    m_data_to_flash.push_back({addrPhy, data});

                    break;

                case 1:
                    // end of hex file
                    //UPDATE_LOG_D("end of hex file";
                    break;

                case 2:
                    m_base_phy_addr = (std::uint32_t)((data.at(0) << 8) + data.at(1)) << 4;
                    m_extended_linear_addr_record = false;
                    break;

                case 3:
                    break;


                case 4:
                    m_base_phy_addr = (std::uint32_t)((data.at(0) << 8) + data.at(1)) << 16;
                    m_extended_linear_addr_record = true;

                    //UPDATE_LOG_D("m_base_phy_addr: 0x" << hozon::netaos::log::LogHex32{m_base_phy_addr};
                    break;
                case 5:
                    break;

                default:
                    break;
            }
        }

        m_bin_file_size = m_flash_end_addr - m_flash_start_addr + 1;

        UM_INFO << "m_flash_start_addr: 0x" << hozon::netaos::log::LogHex32{m_flash_start_addr};
        UM_INFO << "m_flash_end_addr: 0x" << hozon::netaos::log::LogHex32{m_flash_end_addr};
        UM_INFO << "m_bin_file_size: 0x" << hozon::netaos::log::LogHex32{m_bin_file_size};


        std::uint8_t * memory_block = new std::uint8_t[m_bin_file_size];


        for(auto i = m_data_to_flash.begin(); i != m_data_to_flash.end(); i++)
        {
            std::uint32_t memory_block_position = i->startAddr - m_flash_start_addr;
            for (auto j = i->data.begin(); j != i->data.end(); j++)
            {
                memory_block[memory_block_position++] = *j;
            }

        }

        std::ofstream outFile(m_file_out.c_str(), std::ios::out);
        if (!outFile)
        {
            UM_ERROR << "file : " << m_file_out.c_str() << "open error!";
            delete[] memory_block;
            return -13;
        }

        for (std::uint32_t k = 0; k < m_bin_file_size; k++)
        {
            outFile << memory_block[k];
        }

        UM_INFO << "file: [" << m_file_out.c_str() << "] Generated !!!";

        delete[] memory_block;


        return 0;
    }



} // end of update
} // end of netaos
} // end of hozon
/* EOF */
