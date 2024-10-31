
#include "phm/common/include/timer_manager.h"
#include <iostream>
#include <errno.h>

namespace hozon {
namespace netaos {
namespace phm {

std::mutex TimerManager::ins_mtx_;
TimerManager* TimerManager::instance = nullptr;


TimerManager::TimerManager() : epfd_(-1), is_stoped_(false)
{
    memset(events_, 0, sizeof(struct epoll_event) * EPOLL_SIZE);
}

TimerManager*
TimerManager::Instance()
{
    if (nullptr == instance) {
        std::lock_guard<std::mutex> lck(ins_mtx_);
        if (nullptr == instance) {
            instance = new TimerManager();
        }
    }

    return instance;
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
    is_stoped_ = true;
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

    if (instance != nullptr) {
        delete instance;
        instance = nullptr;
    }
}

void TimerManager::Run()
{
    int nfds;
    while (!is_stoped_) {
        nfds = epoll_wait(epfd_, events_, EPOLL_SIZE, 1000);
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
                    std::shared_ptr<timer_context_t> timerTtx = timer_ctx_map_[fd];
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
    if (timerFd > 0) {
        return timerFd;
    }

    if (msTime <= 0) {
        return -1;
    }

    if (timer_ctx_map_.count(timerFd) > 0) {
        return timerFd;
    }

    int tfd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC); // CLOCK_REALTIME or CLOCK_MONOTONIC
    if (tfd < 0) {
        return -1;
    }
    timerFd = tfd;

    struct timespec startTime, intervalTime;
    startTime.tv_sec = msTime / 1000;
    startTime.tv_nsec = (msTime % 1000) * 1000 * 1000;
    intervalTime.tv_sec = msTime / 1000;
    intervalTime.tv_nsec = (msTime % 1000) * 1000 * 1000;

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
        std::shared_ptr<timer_context_t> timerTtx = std::make_shared<timer_context_t>();
        timerTtx->task = task;
        timerTtx->data = data;
        timerTtx->bLoop = bLoop;

        timer_ctx_map_[tfd] = timerTtx;
        // printf("TimerManager::StartFdTimer size: %ld, fd:%d\n", timer_ctx_map_.size(), tfd);
    }

    return 0;
}

int TimerManager::StopFdTimer(int &timerFd)
{
    if (0 >= timerFd) {
        return 0;
    }

    std::lock_guard<std::recursive_mutex> lck(mtx_);
    auto iter = timer_ctx_map_.find(timerFd);
    if(iter == timer_ctx_map_.end()) {
        return -1;
    }

    struct epoll_event event;
    event.data.fd = timerFd;
    event.events = EPOLLOUT;
    if (0 != epoll_ctl(epfd_, EPOLL_CTL_DEL, timerFd, &event)) {
        printf("TimerManager::StopFdTimer strerror:%s\n", strerror(errno));
        return -1;
    }

    if (iter->second) {
        iter->second = nullptr;
        timer_ctx_map_.erase(iter);
    }

    close(timerFd);
    timerFd = -1;
    return 0;
}

void TimerManager::StopAll()
{
    std::lock_guard<std::recursive_mutex> lck(mtx_);
    for (auto &item : timer_ctx_map_) {
        struct epoll_event event;
        event.data.fd = item.first;
        event.events = EPOLLOUT;
        epoll_ctl(epfd_, EPOLL_CTL_DEL, item.first, &event);

        if (item.second) {
            item.second = nullptr;
        }
        close(item.first);
    }

}


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
/* EOF */