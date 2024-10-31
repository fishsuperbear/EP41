/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CanCommandTask implement
 */


#include "update_manager/file_to_bin/file_to_bin.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/file_to_bin/s19_to_bin.h"
#include "update_manager/file_to_bin/hex_to_bin.h"
#include "update_manager/common/common_operation.h"


namespace hozon {
namespace netaos {
namespace update {

    FileToBin::FileToBin()
    {
        m_toBin.clear();
    }

    bool FileToBin::Transition(std::string file_in, std::string bin_file_path)
    {
        UM_DEBUG << "Transition start !";

        if (!PathExists(file_in)){
            UPDATE_LOG_E("input file: %s not Exists", file_in.c_str());
            return false;
        }

        auto it = m_toBin.begin();
        for(; it != m_toBin.end(); it++) {
            if (file_in == it->get()->GetInputFileName()) {
                break;
            }
        }

        if (it != m_toBin.end()) {
            // Already Transition.
            return true;
        }
        else {
            //Haven't Transitio, create a new one.
            auto pos_s19_1 = file_in.find(".s19");
            auto pos_s19_2 = file_in.find(".S19");

            std::string strfileName = getFileName(file_in);
            auto pos_ext = strfileName.find_last_of(".");

            if (pos_ext == std::string::npos)
            {
                UPDATE_LOG_E("input file: %s not have extension", file_in.c_str());
            }

            std::string binFileName = strfileName.substr(0, pos_ext) + std::string(".bin");

            std::unique_ptr<ToBinBase> tmp{nullptr};

            if ((pos_s19_1 != std::string::npos) || (pos_s19_2 != std::string::npos))
            {
                tmp = std::make_unique<S19ToBin>(file_in, bin_file_path + binFileName);
            }
            else
            {
                auto pos_hex_1 = file_in.find(".hex");
                auto pos_hex_2 = file_in.find(".HEX");

                if ((pos_hex_1 != std::string::npos) || (pos_hex_2 != std::string::npos))
                {
                    tmp = std::make_unique<HexToBin>(file_in, bin_file_path + binFileName);
                }
            }

            if (tmp){
                tmp->Transition();
                // Store the new logger to list.
                m_toBin.push_back(std::move(tmp));
            }
        }
        return true;
    }

    std::string FileToBin::GetBinFileName(std::string file_in)
    {
        std::string sRet = "";

        auto it = m_toBin.begin();
        for(; it != m_toBin.end(); it++) {
            if (file_in == it->get()->GetInputFileName()) {
                break;
            }
        }

        if (it != m_toBin.end()) {
            // Found.
            sRet = it->get()->GetBinFileName();
        }

        return sRet;
    }
    std::uint32_t FileToBin::GetBinFileSize(std::string file_in)
    {
        std::uint32_t ret = 0;

        auto it = m_toBin.begin();
        for(; it != m_toBin.end(); it++) {
            if (file_in == it->get()->GetInputFileName()) {
                break;
            }
        }

        if (it != m_toBin.end()) {
            // Found.
            ret = it->get()->GetBinFileSize();
        }

        return ret;
    }
    std::uint32_t FileToBin::GetFlashStartAddr(std::string file_in)
    {
        std::uint32_t ret = 0;

        auto it = m_toBin.begin();
        for(; it != m_toBin.end(); it++) {
            if (file_in == it->get()->GetInputFileName()) {
                break;
            }
        }

        if (it != m_toBin.end()) {
            // Found.
            ret = it->get()->GetFlashStartAddr();
        }

        return ret;
    }
    std::uint32_t FileToBin::GetFlashEndAddr(std::string file_in)
    {
        std::uint32_t ret = 0;

        auto it = m_toBin.begin();
        for(; it != m_toBin.end(); it++) {
            if (file_in == it->get()->GetInputFileName()) {
                break;
            }
        }

        if (it != m_toBin.end()) {
            // Found.
            ret = it->get()->GetFlashEndAddr();
        }

        return ret;
    }


} // end of update
} // end of netaos
} // end of hozon
/* EOF */
