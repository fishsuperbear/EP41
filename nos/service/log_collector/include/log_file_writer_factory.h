// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_writer_factory.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_FILE_WRITER_FACTORY_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_FILE_WRITER_FACTORY_H__

#include "log_collector/include/log_file_writer_base.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class LogFileWriterFactory {
public:
    static LogFileWriterPtr Create(LogFileWriterType log_file_writer_type = LogFileWriterType::FWRITE);
};

}
}
}

#endif // __LOG_COLLECTOR_INCLUDE_LOG_FILE_WRITER_FACTORY_H__
