//
// Created by cheng on 23-6-26.
//

#pragma once
#ifndef TOOLS_THREAD_GUARD_H
#define TOOLS_THREAD_GUARD_H

#include <thread>
#include <vector>
class ThreadsGuard
{
public:
    ThreadsGuard(std::vector<std::thread>& v)
            : threads_(v)
    {
    }

    ~ThreadsGuard()
    {
        for (size_t i = 0; i != threads_.size(); ++i)
        {
            if (threads_[i].joinable())
            {
                threads_[i].join();
            }
        }
    }
private:
    ThreadsGuard(ThreadsGuard&& tg) = delete;
    ThreadsGuard& operator = (ThreadsGuard&& tg) = delete;

    ThreadsGuard(const ThreadsGuard&) = delete;
    ThreadsGuard& operator = (const ThreadsGuard&) = delete;
private:
    std::vector<std::thread>& threads_;
};

#endif //TOOLS_THREAD_GUARD_H
