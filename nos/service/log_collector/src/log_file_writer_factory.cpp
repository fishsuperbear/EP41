// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_writer_factory.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 
/// @date 2023-11-15

#include "log_collector/include/log_file_writer_factory.h"

#include <memory>

#include "log_collector/include/log_file_memcpy_writer.h"
#include "log_collector/include/log_file_fwrite_writer.h"

namespace hozon {
namespace netaos {
namespace logcollector {

LogFileWriterPtr LogFileWriterFactory::Create(LogFileWriterType log_file_writer_type) {
    switch (log_file_writer_type) {
        case LogFileWriterType::FWRITE:
            return std::make_shared<LogFileFwriteWriter>();
        case LogFileWriterType::MEMCPY:
            return std::make_shared<LogFileMemcpyWriter>();
        default:
            return std::make_shared<LogFileMemcpyWriter>();
    }
}

} // namespace logcollector
} // namespace netaos
} // namespace hozon
