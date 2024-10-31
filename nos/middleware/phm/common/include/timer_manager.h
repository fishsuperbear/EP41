/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: Timer implemented with timerfd
 */

#ifndef PHM_TIMER_H
#define PHM_TIMER_H

#include <unistd.h>
#include <functional>
#include <sys/epoll.h>
#include <sys/timerfd.h>
#include <string.h>
#include <unordered_map>
#include <thread>
#include <mutex>


namespace hozon {
namespace netaos {
namespace phm {

const int EPOLL_SIZE = 20;

typedef struct timer_context {
    std::function<void(void*)> task;
    void* data;
    bool bLoop;
} timer_context_t;


class TimerManager
{
public:
    static TimerManager* Instance();

    int Init();
    void DeInit();
    int StartFdTimer(int &timerFd, unsigned int msTime, std::function<void(void*)> task, void* data, bool bLoop = false);
    int StopFdTimer(int &timerFd);
    void StopAll();

private:
    TimerManager();
    TimerManager(const TimerManager &);
    TimerManager & operator = (const TimerManager &);

    void Run();

private:
    static TimerManager* instance;
    static std::mutex ins_mtx_;

    std::thread thread_;
    int epfd_;
    struct epoll_event events_[EPOLL_SIZE];
    std::unordered_map<int, std::shared_ptr<timer_context_t> > timer_ctx_map_;
    bool is_stoped_;
    std::recursive_mutex mtx_;
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_TIMER_H