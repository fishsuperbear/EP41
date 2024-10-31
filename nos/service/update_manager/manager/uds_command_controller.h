#pragma once

#include <stdint.h>
#include <mutex>
#include <string>
#include <atomic>
#include <vector>

namespace hozon {
namespace netaos {
namespace update {

class UdsCommandController {
public:

    static UdsCommandController* Instance();

    void Init();
    void Deinit();
    // 不按照指定顺序处理，则返回false
    bool ProcessCommand(const std::string& command);
    bool ResetProcess();

    bool IsSocVersionSame();
    bool SetVersionSameFlag(bool flag);
private:
    UdsCommandController();
    ~UdsCommandController();
    UdsCommandController(const UdsCommandController &);
    UdsCommandController & operator = (const UdsCommandController &);

    static std::mutex m_mtx;
    static UdsCommandController* m_pInstance;

    std::vector<std::string> expectedCommands;
    uint16_t nextExpectedIndex;
    std::atomic<bool> isVersionSame {false};
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
