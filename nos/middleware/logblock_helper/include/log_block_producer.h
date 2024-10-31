// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_block_producer.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-07

#ifndef __LOG_BLOCK_HELPER_LOG_BLOCK_LOG_BLOCK_WRITER_H__
#define __LOG_BLOCK_HELPER_LOG_BLOCK_LOG_BLOCK_WRITER_H__

#include <string>

#include "log_block_writer.h"

namespace hozon {
namespace netaos {
namespace logblock {

// /*******************************************************
/// @brief It's designed for eath thread caller
// *******************************************************/
class LogBlockProducer {
public:
  static LogBlockProducer& Instance();

public:
  enum class WriteStatusCode : int {
      SUCCESS = 0,             // success
      INIT_FAILED,             // init failed
      RESET_WRITE_POSITION,    // no space to be written
      INVALID_DATA             // invalid data
  };

public:
  // /*******************************************************
  /// @brief Write log to logblock directly.
  ///
  /// @param: str
  /// @param: data_type
  /// @param: version
  ///
  /// @returns: WriteStatusCode 
  // *******************************************************/
  WriteStatusCode Write(const std::string &appid, const std::string &str,
              unsigned int data_type = 0, unsigned short version = 0x1) noexcept;
  WriteStatusCode Write(const std::string &appid, const char *str, int size,
              unsigned int data_type = 0, unsigned short version = 0x1) noexcept;

private:
  bool IsInited() const;
  bool Init(int data_type, const std::string &appid);

private:
  LogBlockProducer(const LogBlockProducer &) = delete;
  LogBlockProducer& operator = (const LogBlockProducer &) = delete;

private:
  LogBlockProducer();
  ~LogBlockProducer();

private:
  static thread_local LogBlockProducer instance_;

  bool inited_ = false;
  LogBlockHandle log_block_handle_ = nullptr;
  LogBlockWriterInfo log_block_writer_info_;
};

} // namespace logblock
} // namespace netaos
} // namespace hozon

#endif // __LOG_BLOCK_HELPER_LOG_BLOCK_LOG_BLOCK_WRITER_H__
