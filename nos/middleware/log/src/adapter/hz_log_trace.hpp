/************************************************************************************
*  Copyright (c) Hozon Technologies Co., Ltd. 2022-2030. All rights reserved.       *
*                                                                                   *
*  @file     hz_log_trace.hpp                                                     *
*  @brief    Define of class HzLogTrace                                          *
*  Details.                                                                         *
*                                                                                   *
*  @version  0.0.0.1                                                                *
*                                                                                   *
*-----------------------------------------------------------------------------------*
*  Change History :                                                                 *
*  <Date>     | <Version> | <Author>       | <Description>                          *
*-----------------------------------------------------------------------------------*
*  2022/06/15 | 0.0.0.1   | YangPeng      | Create file                             *
*-----------------------------------------------------------------------------------*
*                                                                                   *
*************************************************************************************/



#ifndef __HZ_LOG_TRACE_HPP__
#define __HZ_LOG_TRACE_HPP__

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "adapter/hz_stdout_color_sinks.h"
#include "adapter/hz_rotating_file_sink.h"
#include "adapter/log_device_adapter.hpp"
#include "adapter/hz_log_service_sink.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/async.h"

namespace hozon {
namespace netaos {
namespace log {

/**
* @brief HzLogTrace class
* implement of class ILogDeviceAdapter.
*/
class HzLogTrace : public ILogDeviceAdapter
{
 public:
        /** 
        * @brief Constructor function of class HzLogTrace
        *
        */
        HzLogTrace()
        :logFileBaseName_("")
        ,logFilePath_("")
        ,maxLogFileNum_(0)
        ,maxSizeOfLogFile_(0)
        ,serverIp_("")
        ,port_(0)
        ,file_log_name_("hz_log_file")
        ,terminal_log_name_("hz_log_terminal")
        {
        }

        /** 
        * @brief Destructor function of class HzLogTrace
        * 
        */
        ~HzLogTrace() 
        {
            // std::cout << "~HzLogTrace:" << std::endl;

            if (logger_terminal_){
                // std::cout << "~logger_terminal_:" << std::endl;

                spdlog::drop(terminal_log_name_);
                logger_terminal_ = nullptr;
            }

            if (logger_file_){
                // std::cout << "~logger_file_:" << std::endl;
                
                spdlog::drop(file_log_name_);
                logger_file_ = nullptr;
            }
        }

        void setFileName(const std::string& name)
        {
            file_log_name_ = name;
        }

        void setTerminalName(const std::string& name)
        {
            terminal_log_name_ = name;
        }

        /** 
        * @brief Init device function
        * 
        *
        */
        virtual void initDevice(bool pureLogFormat) override
        {
            if (logger_terminal_) {
                if (pureLogFormat) {
                    logger_terminal_->set_pattern("%v");
                }
                else {
                    logger_terminal_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%4!l%$] %v]");
                }

                logger_terminal_->set_level(spdlog::level::trace);
                //logger_terminal_->flush_on(spdlog::level::warn);
            }

            if (logger_file_) {
                if (pureLogFormat) {
                    logger_file_->set_pattern("%v");
                }
                else {
                    logger_file_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%4!l] %v]");
                }

                logger_file_->set_level(spdlog::level::trace);
                logger_file_->flush_on(spdlog::level::err);
            }

            spdlog::flush_every(std::chrono::seconds(1));

            if (logger_service_) {
                if (pureLogFormat) {
                    logger_service_->set_pattern("%v");
                } else {
                    logger_service_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%4!l] %v]");
                }
                logger_service_->set_level(spdlog::level::trace);
            }
        }

        /** 
        * @brief setLog2Terminal function
        * 
        * @param logTo         true or false
        *
        */
        virtual void setLog2Terminal(bool logTo) override
        {
            if (logger_terminal_ != nullptr) {
                spdlog::drop(terminal_log_name_);
                logger_terminal_ = nullptr;
            }

            if(logTo) {
                logger_terminal_ = spdlog::create_async<spdlog::sinks::hz_stdout_color_sink_mt>(terminal_log_name_);
            }
        }

        /** 
        * @brief setLog2File function
        * 
        * @param logTo         true or false
        * @param filePath         file Path
        * @param fileBaseName         file base name
        * @param maxLogFileNum        max Log File Num
        * @param maxSizeOfLogFile        max Size Of Log File
        *
        */
        virtual void setLog2File(bool logTo, std::string filePath, std::string fileBaseName, std::uint32_t maxLogFileNum, std::uint64_t maxSizeOfLogFile) override
        {

            if (logger_file_ != nullptr) {
                spdlog::drop(file_log_name_);
                logger_file_ = nullptr;
            }

            logFileBaseName_ = fileBaseName;
            logFilePath_ = filePath;
            maxLogFileNum_ = maxLogFileNum;
            maxSizeOfLogFile_ = maxSizeOfLogFile;

            if(logTo) {
                logger_file_ = spdlog::create_async<spdlog::sinks::hz_rotating_file_sink_mt>(file_log_name_,
                                                                                         logFilePath_, logFileBaseName_, maxSizeOfLogFile_, maxLogFileNum_);
            }
        }

       /** 
        * @brief setLog2LogService function
        * 
        * @param logTo         true or false
        * @param fileBaseName         file base name
        *
        */
        virtual void setLog2LogService(bool logTo, std::string fileBaseName) override
        {
            logFileBaseName_ = fileBaseName;

            if(logTo) {
                logger_service_ = std::make_shared<HzLogServiceSink>(fileBaseName);
            }
        }

        /** 
        * @brief setLog2Remote function
        * 
        * @param logTo         true or false
        * @param serverIp         server Ip
        * @param port        port
        *
        */
        virtual void setLog2Remote(bool logTo, std::string serverIp, std::uint32_t port) override
        {
            if (logTo){
                serverIp_ = serverIp;
                port_ = port;
            }
        }

        /** 
        * @brief critical c++ stype function implement
        * 
        * @param message ... The message to be output
        *
        */
        virtual void critical(const std::string& message) override
        {
            if (logger_terminal_) {
                logger_terminal_->critical(message);
            }

            if (logger_file_) {
                logger_file_->critical(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::critical, message);
            }
        }
        
        /** 
        * @brief error c++ stype function implement
        * 
        * @param message ... The message to be output
        *
        */
        virtual void error(const std::string& message) override
        {
            if (logger_terminal_) {
                logger_terminal_->error(message);
            }

            if (logger_file_) {
                logger_file_->error(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::err, message);
            }
        }


        /** 
        * @brief warn c++ stype function implement
        * 
        * @param message ... The message to be output
        *
        */
        virtual void warn(const std::string& message) override
        {
            if (logger_terminal_) {
                logger_terminal_->warn(message);
            }

            if (logger_file_) {
                logger_file_->warn(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::warn, message);
            }
        }

        /** 
        * @brief info c++ stype function implement
        * 
        * @param message ... The message to be output
        *
        */
        virtual void info(const std::string& message) override
        {
            if (logger_terminal_) {
                logger_terminal_->info(message);
            }

            if (logger_file_) {
                logger_file_->info(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::info, message);
            }
        }

        
        /** 
        * @brief debug c++ stype function implement
        * 
        * @param message ... The message to be output
        *
        */
        virtual void debug(const std::string& message) override
        {
            if (logger_terminal_) {
                logger_terminal_->debug(message);
            }

            if (logger_file_) {
                logger_file_->debug(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::debug, message);
            }
        }


        /** 
        * @brief trace c++ stype function implement
        * 
        * @param message ... The message to be output
        *
        */
        virtual void trace(const std::string& message) override
        {
            if (logger_terminal_) {
                logger_terminal_->trace(message);
            }

            if (logger_file_) {
                logger_file_->trace(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::trace, message);
            }
        }


 private:
        
        std::shared_ptr<spdlog::logger> logger_file_ {nullptr};
        std::shared_ptr<spdlog::logger> logger_terminal_ {nullptr};
        std::shared_ptr<HzLogServiceSink> logger_service_ {nullptr};
        std::string logFilePath_; 
        std::string logFileBaseName_; 
        std::uint64_t maxLogFileNum_;
        std::uint64_t maxSizeOfLogFile_;
        std::string serverIp_;
        std::uint32_t port_;
        std::string file_log_name_;
        std::string terminal_log_name_;
};


}
}
}
#endif  // __HZ_LOG_TRACE_HPP__ 
