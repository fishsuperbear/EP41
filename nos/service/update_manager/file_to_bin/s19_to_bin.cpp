/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CanCommandTask implement
 */

#include "update_manager/file_to_bin/s19_to_bin.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/common/common_operation.h"

#include <fstream>
#include <string>
#include <iostream>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <iostream>

namespace hozon {
namespace netaos {
namespace update {


    S19ToBin::S19ToBin(std::string s19_file_in, std::string bin_file_out)
        :ToBinBase()
        ,m_bin_file_size(0)
        ,m_flash_start_addr(0xFFFFFFFF)
        ,m_flash_end_addr(0)
        ,m_file_in(s19_file_in)
        ,m_file_out(bin_file_out)
    {
        m_data_to_flash.clear();


    }

    std::int32_t S19ToBin::parseLine(const std::uint8_t & addrLength, const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        //UPDATE_LOG_D(" S19ToBin::parseLine start!";
        std::uint8_t start_pos = 2;
        std::uint8_t check_sum_count = 0;
        count = std::stoi(line.substr(start_pos, 2), 0, 16);

        addr = static_cast<std::uint32_t>(std::stoul(line.substr(start_pos + 2, addrLength), 0, 16));



        if ((2 * count) != line.length() - 4 -1)
        {
            UM_ERROR << "line data error!";
            UM_ERROR<< "count :" << static_cast<uint16_t>(count) << ", line.length() - 4: " << line.length() -4;
            return -1;
        }

        check_sum_count = count + ((addr >> 24) & 0xFF) + ((addr >> 16) & 0xFF) + ((addr >> 8) & 0xFF) + (addr  & 0xFF);

        std::string strData = line.substr(start_pos + 2 + addrLength, (2 * count) - 2 - addrLength);
        check_sum = std::stoi(line.substr(line.length() - 1 - 2, 2), 0, 16);

        //UPDATE_LOG_D("count:" << static_cast<uint16_t>(count) << ", addr:" << addr << ", check_sum:" << static_cast<uint16_t>(check_sum) << ", strData:" << strData.c_str();

        data.clear();

        for (std::uint16_t i = 0; (2 * i) < strData.size(); i++)
        {
            std::uint8_t tmpData = std::stoi(strData.substr((2 * i), 2), 0, 16);
            data.push_back(tmpData);
            check_sum_count = (check_sum_count + tmpData) & 0xFF;
        }

        if ((std::uint8_t)(check_sum) != (std::uint8_t)(0xff - check_sum_count))
        {
            UM_ERROR << "S19ToBin::parseLine check sum error, check_sum_count:" << check_sum_count << " , read check_sum:" << check_sum;
            return -2;
        }

        return 0;
    }

    std::int32_t S19ToBin::parseS0Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(4, line, count, addr, data, check_sum);
    }
    std::int32_t S19ToBin::parseS1Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(4, line, count, addr, data, check_sum);

    }
    std::int32_t S19ToBin::parseS2Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(6, line, count, addr, data, check_sum);

    }
    std::int32_t S19ToBin::parseS3Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(8, line, count, addr, data, check_sum);

    }
    std::int32_t S19ToBin::parseS4Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return 0;
    }
    std::int32_t S19ToBin::parseS5Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(4, line, count, addr, data, check_sum);

    }
    std::int32_t S19ToBin::parseS6Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(6, line, count, addr, data, check_sum);

    }
    std::int32_t S19ToBin::parseS7Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(8, line, count, addr, data, check_sum);

    }
    std::int32_t S19ToBin::parseS8Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(6, line, count, addr, data, check_sum);

    }
    std::int32_t S19ToBin::parseS9Line(const std::string & line, std::uint8_t & count, std::uint32_t & addr, std::vector<std::uint8_t> & data, std::uint8_t & check_sum)
    {
        return parseLine(4, line, count, addr, data, check_sum);

    }

    std::int32_t S19ToBin::Transition()
    {
        auto pos_s19 = m_file_in.find(".s19");

        if (pos_s19 == std::string::npos)
        {
            pos_s19 = m_file_in.find(".S19");
            if (pos_s19 == std::string::npos)
            {
                UM_ERROR << "file : " << m_file_in.c_str() << "not a s19 file!";
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
        std::string strS_Icon;
        std::uint8_t type = 0;
        std::uint8_t count = 0;
        std::uint32_t addr = 0;
        std::uint8_t check_sum = 0;
        std::int32_t ret = -1;

        while(std::getline(inFile, strline))
        {
            std::vector<std::uint8_t> data;

            if (strline.length() < 4)
            {
                UM_ERROR << "the length of line error!";
                return -2;
            }

            //UPDATE_LOG_D("strline: " << strline.c_str();

            strS_Icon = strline.substr(0, 1);
            type = std::stoi(strline.substr(1, 1), 0, 16);
            count = std::stoi(strline.substr(2, 2), 0, 16);

            //UPDATE_LOG_D("strS_Icon: " << strS_Icon.c_str() << "type: " << type  << "count: "<< static_cast<uint16_t>(count);

            data.clear();

            if (strS_Icon != std::string("S") && strS_Icon != std::string("s"))
            {
                UM_ERROR << "Not start with S or s!";
                return -3;
            }

            switch (type)
            {
                case 0:

                    ret = parseS0Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS0Line error!";
                        return -4;
                    }

                    break;

                case 1:
                    ret = parseS1Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS1Line error!";
                        return -5;
                    }

                    m_flash_start_addr = m_flash_start_addr < addr ? m_flash_start_addr : addr;
                    m_flash_end_addr = m_flash_end_addr > (addr + data.size() -1) ?  m_flash_end_addr : (addr + data.size()  -1);

                    m_data_to_flash.push_back({addr, data});
                    break;

                case 2:
                    ret = parseS2Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS2Line error!";
                        return -6;
                    }

                    m_flash_start_addr = m_flash_start_addr < addr ? m_flash_start_addr : addr;
                    m_flash_end_addr = m_flash_end_addr > (addr + data.size()  -1) ?  m_flash_end_addr : (addr + data.size()  -1);

                    m_data_to_flash.push_back({addr, data});
                    break;

                case 3:
                    ret = parseS3Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS3Line error!";
                        return -7;
                    }

                    m_flash_start_addr = m_flash_start_addr < addr ? m_flash_start_addr : addr;
                    m_flash_end_addr = m_flash_end_addr > (addr + data.size()  -1) ?  m_flash_end_addr : (addr + data.size()  -1);

                    m_data_to_flash.push_back({addr, data});
                    break;

                case 5:
                    ret = parseS5Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS5Line error!";
                        return -8;
                    }
                    break;
                case 6:
                    ret = parseS6Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS6Line error!";
                        return -9;
                    }
                    break;

                case 7:
                    ret = parseS7Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS7Line error!";
                        return -10;
                    }
                    break;

                case 8:
                    ret = parseS8Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS8Line error!";
                        return -11;
                    }
                    break;

                case 9:
                    ret = parseS9Line(strline, count, addr, data, check_sum);
                    if (0 != ret)
                    {
                        UM_ERROR << "parseS9Line error!";
                        return -12;
                    }
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
