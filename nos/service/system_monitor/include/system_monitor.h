
#ifndef SYSTEM_MONITOR_H
#define SYSTEM_MONITOR_H

namespace hozon {
namespace netaos {
namespace system_monitor {

class SystemMonitor {

public:
    SystemMonitor();
    ~SystemMonitor();

    void Init();
    void DeInit();

    void Run();
    void Stop();

private:
    SystemMonitor(const SystemMonitor &);
    SystemMonitor & operator = (const SystemMonitor &);

private:
    bool stop_flag_;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_H