/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef STATISTICS_DEFIN_H
#define STATISTICS_DEFIN_H
#pragma once

#include <atomic>
#include <map>
#include <string>

namespace hozon {
namespace netaos {
namespace network_capture {



class Counter {
   private:
    // std::map<std::string, std::atomic<int>> count_map_;
    // std::atomic<int> count;
    static Counter* instance_;

   public:
    Counter();
    ~Counter();
    static Counter& Instance();

    // 原子增加计数
    void increment(const std::string&& name);
    void increment(std::atomic<std::uint64_t>& count);

    // 原子减少计数
    void decrement(const std::string&& name);
    void decrement(std::atomic<std::uint64_t>& count);

    // 原子获取当前计数
    const int getCount(const std::string&& name);
    std::uint64_t getCount(std::atomic<std::uint64_t>& count);

    // 统计清零
    void setZero();
    void setZero(std::atomic<std::uint64_t>& count);

    // const std::unordered_map<std::string, std::atomic<int>>& getmap();

    // 打印统计信息
    void printStat(int64_t gap);
};
}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif
