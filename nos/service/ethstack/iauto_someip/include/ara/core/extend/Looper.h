#ifndef ARA_CORE_EXTEND_LOOPER_H
#define ARA_CORE_EXTEND_LOOPER_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <memory>
#include <functional>

namespace ara {
namespace core {
namespace extend {

class LooperContext : public std::enable_shared_from_this<LooperContext> {
public:
    virtual ~LooperContext() = default;
    virtual int submit(std::function<void(void)>&& task) = 0;
};
class MainlooperContext : public LooperContext  {
public:
    virtual ~MainlooperContext() = default;
    virtual int drain_one() = 0;
    virtual int drain_all() = 0;
    virtual int wait(uint64_t time_ms) = 0;
    virtual void run() = 0;
    virtual void quit() = 0;
};
class ThreadPoolContext : public LooperContext {
public:
    virtual ~ThreadPoolContext() = default;
    virtual int start() = 0;
    virtual void stop() = 0;
};

class LooperTimerSource : public std::enable_shared_from_this<LooperTimerSource> {
public:
    virtual ~LooperTimerSource() = default;
    static uint64_t now();
    virtual int start(uint64_t start_ms, uint64_t interval_ms) = 0;
    virtual void stop() = 0;
};
class LooperQueue : public std::enable_shared_from_this<LooperQueue>  {
public:
    virtual ~LooperQueue() = default;
    static LooperQueue* myself();
    virtual int async(std::function<void(void)>&& task) = 0;
    virtual int barrier(std::function<void(void)>&& task) = 0;
    virtual void clear() = 0;
    virtual std::shared_ptr<LooperTimerSource> makeTimerSource(std::function<void(void)>&& task) = 0;
};

std::shared_ptr<MainlooperContext> makeMainlooperContext();
std::shared_ptr<ThreadPoolContext> makeThreadPoolContext(uint32_t min_thread, uint32_t max_thread, uint32_t priority, uint64_t keep_alive_time_ms, const char* name = "unknown");
std::shared_ptr<LooperQueue> makeLooperQueue(const std::shared_ptr<LooperContext>& context, bool is_concurrent, uint32_t max_queue_size, const char* name = "");

}  // namespace extend
}  // namespace core
}  // namespace ara

#endif // ARA_CORE_EXTEND_LOOPER_H
/* EOF */
