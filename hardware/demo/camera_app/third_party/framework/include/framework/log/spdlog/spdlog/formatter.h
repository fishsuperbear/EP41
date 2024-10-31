#pragma once

#include <spdlog/fmt/fmt.h>
#include <spdlog/details/log_msg.h>

namespace spdlog {

class formatter
{
public:
    virtual ~formatter() = default;
    virtual void format(const details::log_msg &msg, memory_buf_t &dest) = 0;
    virtual std::unique_ptr<formatter> clone() const = 0;
};
} // namespace spdlog
