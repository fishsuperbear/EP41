#pragma once

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
namespace update {

const int EPOLL_SIZE = 20;

typedef struct timer_context {
    std::function<void(void*)> task;
    void* data;
    bool bLoop;
} timer_context_t;


class TimerManager
{
public:
    TimerManager();
    ~TimerManager();

    int Init();
    void DeInit();
    int StartFdTimer(int &timerFd, unsigned int msTime, std::function<void(void*)> task, void* data, bool bLoop = false);
    int StopFdTimer(int &timerFd);
    void StopAll();

private:
    TimerManager(const TimerManager &);
    TimerManager & operator = (const TimerManager &);

    void Run();

private:
    std::thread thread_;
    int epfd_;
    struct epoll_event events_[EPOLL_SIZE];
    std::unordered_map<int, timer_context_t*> timerCtxMap_;
    bool isStoped;
    std::recursive_mutex mtx_;
};


}  // namespace update
}  // namespace netaos
}  // namespace hozon