#include "cm/include/proxy.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "log/include/default_logger.h"
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <condition_variable>
#include <mutex>
#include <csignal>

using namespace hozon::netaos::cm;

class FreqChecker {
    using checker_time = std::chrono::time_point<std::chrono::system_clock>;

   public:
    FreqChecker() = default;
    void say(const std::string& unique_name, uint64_t sample_cnt = 100);

   private:
    std::unordered_map<std::string, std::pair<uint64_t, checker_time>> freq_map_;
};

void FreqChecker::say(const std::string& unique_name, uint64_t sample_cnt) {
    if (freq_map_.find(unique_name) == freq_map_.end()) {
        freq_map_[unique_name] = std::make_pair(1, std::chrono::system_clock::now());
    } else {
        freq_map_[unique_name].first++;
    }

    if (freq_map_[unique_name].first == sample_cnt) {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = now - freq_map_[unique_name].second;
        DF_LOG_INFO << "check " << unique_name << " frequency: " << sample_cnt / diff.count() << " Hz";
        freq_map_[unique_name].second = now;
        freq_map_[unique_name].first = 0;
    }
}

class ProtoCMProxy {
public:
    ProtoCMProxy() : 
            _proxy(std::make_shared<CmProtoBufPubSubType>()) {
    }

    ~ProtoCMProxy() {

    }

    using CallbackFunc = std::function<void(void)>;
    int32_t Init(const uint32_t domain, const std::string& topic, CallbackFunc cb) {
        int32_t ret = _proxy.Init(domain, topic);
        if (ret < 0) {
            return ret;
        }

        _proxy.Listen(std::bind(&ProtoCMProxy::ProxyListenCallback, this, topic));

        return 0;
    }

    void Deinit() {
        _proxy.Deinit();
    }

private:
    void ProxyListenCallback(const std::string& topic) {
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);
        _proxy.Take(cm_pb);

        checker.say(topic);
    }

    Proxy _proxy;
    CallbackFunc _cb;
    FreqChecker checker;
};

void cb(void) {}

bool g_stopFlag = false;
std::mutex mtx;
std::condition_variable cv;

void INTSigHandler(int32_t num)
{
    (void)num;
    g_stopFlag = true;
    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_all();
}

int main(int argc, char* argv[]) {
    /*Need add SIGTERM from EM*/
    signal(SIGTERM, INTSigHandler);
    signal(SIGINT, INTSigHandler);
    signal(SIGPIPE, SIG_IGN);

    ProtoCMProxy reader;
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret = reader.Init(0, std::string(argv[1]), cb);

    while (!g_stopFlag) {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck);
    }

    reader.Deinit();
    DF_LOG_INFO << "Deinit end." << ret;
}