
#include "diag/common/include/timer_manager.h"


namespace hozon {
namespace netaos {
namespace diag {


TimerManager::TimerManager() : epfd_(-1), isStoped(false)
{
    memset(events_, 0, sizeof(struct epoll_event) * EPOLL_SIZE);
}

TimerManager::~TimerManager()
{

}

int TimerManager::Init()
{
    if (epfd_ > 0) {
        return 0;
    }

    epfd_ = epoll_create(EPOLL_SIZE);
    if (epfd_ < 0) {
        return -1;
    }

    thread_ = std::thread(&TimerManager::Run, this);

    return 0;
}

void TimerManager::DeInit()
{
    isStoped = true;
    StopAll();

    uint8_t retry_count = 0;
    do {
        int tfd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
        if (tfd < 0) {
            continue;
        }

        struct timespec startTime, intervalTime;
        startTime.tv_sec = 0;
        startTime.tv_nsec = 1000 * 1000;
        intervalTime.tv_sec = 0;
        intervalTime.tv_nsec = 1000 * 1000;

        struct itimerspec newValue;
        newValue.it_value = startTime;
        newValue.it_interval = intervalTime;

        if (timerfd_settime(tfd, 0, &newValue, NULL) < 0) {
            close(tfd);
            continue;
        }

        struct epoll_event event;
        event.data.fd = tfd;
        event.events = EPOLLIN | EPOLLET;

        if (epoll_ctl(epfd_, EPOLL_CTL_ADD, tfd, &event) < 0) {
            close(tfd);
            continue;
        }

        break;
    } while (retry_count++ < 10);

    if (retry_count < 10) {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    if (epfd_ > 0) {
        close(epfd_);
        epfd_ = -1;
    }
}

void TimerManager::Run()
{
    pthread_setname_np(pthread_self(), "timer_run");
    int nfds;
    while (!isStoped) {

        nfds = epoll_wait(epfd_, events_, EPOLL_SIZE, -1);
        if (nfds == 0) {
            continue;
        }

        for (int i = 0; i < nfds; i++) {
            if (events_[i].events & EPOLLIN) {
                uint64_t data;
                ssize_t res = read(events_[i].data.fd, &data, sizeof(uint64_t));
                if (res <= 0) {
                    continue;
                }
                
                std::lock_guard<std::recursive_mutex> lck(mtx_);
                auto fd = events_[i].data.fd;
                {
                    timer_context_t* timerTtx = timerCtxMap_[fd];
                    if (timerTtx == nullptr) {
                        continue;
                    }

                    timerTtx->task(timerTtx->data);

                    if (!timerTtx->bLoop) {
                        StopFdTimer((int&)fd);
                    }
                }
            }
        }
    }

    if (epfd_ > 0) {
        close(epfd_);
        epfd_ = -1;
    }
}

int TimerManager::StartFdTimer(int &timerFd, unsigned int msTime, std::function<void(void*)> task, void* data, bool bLoop)
{
    // printf("TimerManager::StartFdTimer enter timer fd is %d  msTime is %d\n", timerFd, msTime);
    if (timerFd > 0) {
        return timerFd;
    }

    int tfd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC); // CLOCK_REALTIME or CLOCK_MONOTONIC
    if (tfd < 0) {
        printf("TimerManager::StartFdTimer timerfd_create failed, timer fd is %d  msTime is %d\n tfd is %d", timerFd, msTime, tfd);
        return -1;
    }
    timerFd = tfd;

    struct timespec startTime, intervalTime;
    startTime.tv_sec = msTime / 1000;
    startTime.tv_nsec = (msTime % 1000) * 1000 * 1000;
    intervalTime.tv_sec = msTime / 1000;
    intervalTime.tv_nsec = (msTime % 1000) * 1000 * 1000;

    // printf("TimerManager::StartFdTimer timerfd_create... timer fd is %d tv_sec is %d tv_nsec is %d \n", timerFd, int(startTime.tv_sec), int(startTime.tv_nsec));

    struct itimerspec newValue;
    newValue.it_value = startTime;
    newValue.it_interval = intervalTime;

    if (timerfd_settime(tfd, 0, &newValue, NULL) < 0) {
        return -1;
    }

    struct epoll_event event;
    event.data.fd = tfd;
    event.events = EPOLLIN | EPOLLET;

    if (epoll_ctl(epfd_, EPOLL_CTL_ADD, tfd, &event) < 0) {
        return -1;
    }

    {
        std::lock_guard<std::recursive_mutex> lck(mtx_);
        timer_context_t* timerTtx = new timer_context_t;
        timerTtx->task = task;
        timerTtx->data = data;
        timerTtx->bLoop = bLoop;

        timerCtxMap_[tfd] = timerTtx;
    }

    return 0;
}

int TimerManager::StopFdTimer(int &timerFd)
{
    // printf("TimerManager::StopFdTimer enter timer fd is %d \n", timerFd);

    if (timerFd < 0) {
        return 0;
    }

    std::lock_guard<std::recursive_mutex> lck(mtx_);
    std::unordered_map<int, timer_context_t*>::iterator iter = timerCtxMap_.find(timerFd);
    if(iter == timerCtxMap_.end()) {
        return -1;
    }

    struct epoll_event event;
    event.data.fd = timerFd;
    event.events = EPOLLOUT;
    epoll_ctl(epfd_, EPOLL_CTL_DEL, timerFd, &event);

    if (iter->second) {
        delete iter->second;
        iter->second = nullptr;
    }
    timerCtxMap_.erase(iter);

    close(timerFd);
    timerFd = -1;
    // printf("TimerManager::StopFdTimer close.... timer fd is %d \n", timerFd);

    return 0;
}

void TimerManager::StopAll()
{
    std::lock_guard<std::recursive_mutex> lck(mtx_);
    for (auto &item : timerCtxMap_) {
        struct epoll_event event;
        event.data.fd = item.first;
        event.events = EPOLLOUT;
        epoll_ctl(epfd_, EPOLL_CTL_DEL, item.first, &event);

        if (item.second) {
            delete item.second;
            item.second = nullptr;
        }
        close(item.first);
    }

    timerCtxMap_.clear();
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */