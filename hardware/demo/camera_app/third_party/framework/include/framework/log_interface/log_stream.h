#pragma once

#include <ostream>
#include <type_traits>

#include "framework/log_interface/common.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/spdlog.h"

namespace netaos {
namespace framework {
namespace loginterface {

using neta_memory_buffer =
    fmt::basic_memory_buffer<char, 30 * 1024>;  // log buf reserved for 30kb

class LogStream {
 public:
  SPDLOG_ALWAYS_INLINE
  LogStream() { _buffer.clear(); }

  SPDLOG_ALWAYS_INLINE
  LogStream& operator<<(const char* msg) {
    _buffer.append(msg, msg + std::strlen(msg));
    return *this;
  }

  template <typename T>
  SPDLOG_ALWAYS_INLINE LogStream& operator<<(T* msg) {
    fmt::format_to(std::back_inserter(_buffer), "{}", &msg);
    return *this;
  }

  SPDLOG_ALWAYS_INLINE
  LogStream& operator<<(const spdlog::string_view_t& msg) {
    _buffer.append(msg.begin(), msg.end());
    return *this;
  }

  // T cannot be statically converted to string_view
  template <class T,
            typename std::enable_if<
                !std::is_convertible<const T&, spdlog::string_view_t>::value,
                T>::type* = nullptr>
  SPDLOG_ALWAYS_INLINE LogStream& operator<<(const T& msg) {
    fmt::format_to(std::back_inserter(_buffer), "{}", msg);
    return *this;
  }

  SPDLOG_ALWAYS_INLINE
  LogStream& operator<<(std::ostream& (*func)(std::ostream&)) {
    if ((void*)func == (void*)(std::endl<char, std::char_traits<char>>)) {
      const char* s = "\n";
      _buffer.append(s, s + std::strlen(s));
    }
    // ignore other
    return *this;
  }

  const neta_memory_buffer& buffer() const { return _buffer; }

 private:
  // fmt::memory_buffer _buffer;
  static thread_local neta_memory_buffer _buffer;
};

class LogMessageVoidify {
 public:
  LogMessageVoidify() = default;
  void operator&(LogStream&) {}
};

}  // namespace loginterface
}  // namespace framework
}  // namespace netaos
