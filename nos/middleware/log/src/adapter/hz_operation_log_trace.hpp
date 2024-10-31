#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include "adapter/hz_log_service_sink.hpp"

namespace hozon {
namespace netaos {
namespace log {

class HzOperationLogTrace
{
 public:
        HzOperationLogTrace()
        :logFilePath_("")
        ,maxLogFileNum_(0)
        ,maxSizeOfLogFile_(0)
        ,file_log_name_("hz_operation_log_file")
        {
        }

        ~HzOperationLogTrace() 
        {
            if (logger_file_){
                spdlog::drop(file_log_name_);
                logger_file_ = nullptr;
            }
        }

        void initDevice()
        {
            if (logger_file_) {

                logger_file_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%4!l%$] %v]");
                logger_file_->set_level(spdlog::level::trace);
            }
            spdlog::flush_every(std::chrono::seconds(1));

            if (logger_service_) {
                logger_service_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%4!l%$] %v]");
                logger_service_->set_level(spdlog::level::trace);
            }
        }

        
        void setLog2File(std::string logFileName, std::string filePath, std::uint32_t maxLogFileNum, std::uint64_t maxSizeOfLogFile)
        {

            if (logger_file_ != nullptr) {
                spdlog::drop(file_log_name_);
                logger_file_ = nullptr;
            }
            file_log_name_ = logFileName;
            logFilePath_ = filePath;
            maxSizeOfLogFile_ = maxSizeOfLogFile;
            maxLogFileNum_ = maxLogFileNum;
            /** 
            * 
            * @param file_log_name_ spdLog 实例的名字，对于不同文件，需要以不同的name进行命名
            * @param logFilePath_ 文件名
            *
            */
            logger_file_ = spdlog::create_async<spdlog::sinks::rotating_file_sink_mt>(file_log_name_, logFilePath_, maxSizeOfLogFile_, maxLogFileNum_);
        }

        void setLog2LogService(const std::string &file_base_name) {
            file_log_name_ = file_base_name;
            logger_service_ = std::make_shared<HzLogServiceSink>(file_base_name);
        }

        void critical_log(const std::string& message)
        {
            if (logger_file_) {
                logger_file_->critical(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::critical, message);
            }
        }

        void error_log(const std::string& message)
        {
            if (logger_file_) {
                logger_file_->error(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::err, message);
            }
        }

        void warn_log(const std::string& message)
        {
            if (logger_file_) {
                logger_file_->warn(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::warn, message);
            }
        }

        void info_log(const std::string& message)
        {
            if (logger_file_) {
                logger_file_->info(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::info, message);
            }
        }

        void debug_log(const std::string& message)
        {
            if (logger_file_) {
                logger_file_->debug(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::debug, message);
            }
        }

        void trace_log(const std::string& message)
        {
            if (logger_file_) {
                logger_file_->trace(message);
            }

            if (logger_service_) {
                logger_service_->log(spdlog::level::trace, message);
            }
        }

 private:
        std::shared_ptr<spdlog::logger> logger_file_ {nullptr};
        std::shared_ptr<HzLogServiceSink> logger_service_ {nullptr};
        std::string logFilePath_; 
        std::uint64_t maxLogFileNum_;
        std::uint64_t maxSizeOfLogFile_;
        std::string file_log_name_;
};


}
}
}
