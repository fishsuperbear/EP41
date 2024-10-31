#include "pub.h"
#include "sub.h"

#include <vector>
#include <memory>
#include <unistd.h>
#include "gen/latPubSubTypes.h"
#include "gen/lat.h"
#include <getopt.h>
#include <fstream>

void SetScheduler() {
    sched_param param;
    param.sched_priority = 99;

    sched_setscheduler(0, SCHED_RR, &param);
}

void Usage() {
    LAT_LOG_INFO << "Please specify flow type and number.";
}

double GetCurrTimeStampUs() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);

    return time.tv_sec * 1000 * 1000 + time.tv_nsec / 1000;
}
// std::string("/lat") + std::to_string(index)
class SubInfo {
public:
    SubInfo(const std::string& topic, uint32_t samples, const std::string& type, int32_t ignore) {
        _type = type;

        subscriber = std::make_shared<Subscriber>();
        if (type == "fix1k") {
            subscriber->Init(topic, std::make_shared<Lat1KPubSubType>(), std::bind(&SubInfo::Callback<Lat1K>, this));
        }
        else if (type == "fix10k") {
            subscriber->Init(topic, std::make_shared<Lat10KPubSubType>(), std::bind(&SubInfo::Callback<Lat10K>, this));
        }
        else if (type == "fix100k") {
            subscriber->Init(topic, std::make_shared<Lat100KPubSubType>(), std::bind(&SubInfo::Callback<Lat100K>, this));
        }
        else if (type == "fix200k") {
            subscriber->Init(topic, std::make_shared<Lat200KPubSubType>(), std::bind(&SubInfo::Callback<Lat200K>, this));
        }
        else if (type == "fix500k") {
            subscriber->Init(topic, std::make_shared<Lat500KPubSubType>(), std::bind(&SubInfo::Callback<Lat500K>, this));
        }
        else if (type == "fix1000k") {
            subscriber->Init(topic, std::make_shared<Lat1000KPubSubType>(), std::bind(&SubInfo::Callback<Lat1000K>, this));
        }
        else if (type == "fix16m") {
            subscriber->Init(topic, std::make_shared<Lat16mPubSubType>(), std::bind(&SubInfo::Callback<Lat16m>, this));
        }
        else if (type == "var1k") {
            subscriber->Init(topic, std::make_shared<LatVarPubSubType>(), std::bind(&SubInfo::Callback<LatVar>, this));
        }
        else if (type == "var10k") {
            subscriber->Init(topic, std::make_shared<LatVarPubSubType>(), std::bind(&SubInfo::Callback<LatVar>, this));
        }
        else if (type == "var100k") {
            subscriber->Init(topic, std::make_shared<LatVarPubSubType>(), std::bind(&SubInfo::Callback<LatVar>, this));
        }
        else if (type == "var200k") {
            subscriber->Init(topic, std::make_shared<LatVarPubSubType>(), std::bind(&SubInfo::Callback<LatVar>, this));
        }
        else if (type == "var500k") {
            subscriber->Init(topic, std::make_shared<LatVarPubSubType>(), std::bind(&SubInfo::Callback<LatVar>, this));
        }
        else if (type == "var1000k") {
            subscriber->Init(topic, std::make_shared<LatVarPubSubType>(), std::bind(&SubInfo::Callback<LatVar>, this));
        }
        else {
            LAT_LOG_ERROR << "Unknown type " << type; 
        }

        _samples = samples;
        _topic = topic;
        _ignore = ignore;
        _head_ignore = ignore;
    }

    template<typename T>
    void Callback() {
        std::shared_ptr<T> msg(new T);
        int32_t ret = subscriber->Take(msg.get());
        if (ret < 0) {
            return;
        }

        if (_head_ignore > 0) {
            _head_ignore--;
            return;
        }
    
        _latencies.emplace_back(GetCurrTimeStampUs() - msg->lat().timestamp_us());
        // LAT_LOG_INFO << "latency " << _latencies.back() << "(us).";

        if (last_seq == -1) {
            last_seq = msg->lat().seq();
        }
        else if (msg->lat().seq() != (last_seq + 1)) {
            lost_cnt += (msg->lat().seq() - last_seq - 1);
            LAT_LOG_WARN << "Lost sample, last seq " << last_seq << ", curr seq " << msg->lat().seq();
            last_seq = msg->lat().seq();
        }
        else {
            last_seq = msg->lat().seq();
        }

        if (_latencies.size() == (_samples + _ignore)) {
            Stop();
        }

        // if ((_latencies.size() % 1000) == 0) {
        //     LAT_LOG_INFO << "Recv " << _latencies.size() << " samples";
        // }
    };

    void Stop() {
        subscriber->Stop();
        std::vector<uint64_t> latencies = _latencies;
        latencies.resize(_samples);
        std::sort(latencies.begin(), latencies.end());
        uint64_t mid = latencies[latencies.size() / 2];

        _printlk.lock();
        LAT_LOG_INFO << "[" << _topic << "] " << latencies.size() << " samples, latency min: " 
                << latencies[0] << "(us),  mid: " 
                << mid << "(us), @99.9: " << latencies[latencies.size() * 0.999] << "(us), max: "
                << latencies.back()  << "(us), lost sample:"
                << lost_cnt;

        if ((!_file_name.empty()) && (!_test_name.empty())) {
            WriteResult(latencies[0], mid, latencies[latencies.size() * 0.999], latencies.back(), lost_cnt);
        }
        _printlk.unlock();
        _stop = true;
    }

    void WriteResult(uint64_t min_us, uint64_t mid_us, uint64_t at99d9_us, uint64_t max_us, uint64_t lost_cnt) {
        std::ofstream f(_file_name, std::ios::out | std::ios::app);
        if (!f.is_open()) {
            LAT_LOG_ERROR << "Fail to open file " << _file_name;
            return;
        }

        f << _type << ", " << min_us << ", " << mid_us << ", " << at99d9_us << ", " << max_us << ", " << lost_cnt << std::endl;
    }

    std::string _topic;
    std::shared_ptr<Subscriber> subscriber;
    uint32_t seq = 0;
    std::vector<uint64_t> _latencies;
    uint32_t lost_cnt = 0;
    int32_t last_seq = -1;
    uint32_t _samples = 0;
    bool _stop = false;
    int32_t _ignore;
    int32_t _head_ignore;
    std::string _file_name;
    std::string _test_name;
    std::string _type;

    static std::mutex _printlk;
};
std::mutex SubInfo::_printlk;

int main(int argc, char* argv[]) {
    SetScheduler();
    std::vector<std::shared_ptr<SubInfo>> subs;
    std::string file_name;
    std::string test_name;

    struct option long_opts[] = {
        {"samples",     required_argument,        nullptr,    's'},
        {"fix1k",       required_argument,        nullptr,    1},
        {"fix10k",      required_argument,        nullptr,    2},
        {"fix100k",     required_argument,        nullptr,    3},
        {"fix200k",     required_argument,        nullptr,    4},
        {"fix500k",     required_argument,        nullptr,    5},
        {"fix1000k",    required_argument,        nullptr,    6},
        {"var1k",       required_argument,        nullptr,    7},
        {"var10k",      required_argument,        nullptr,    8},
        {"var100k",     required_argument,        nullptr,    9},
        {"var200k",     required_argument,        nullptr,    10},
        {"var500k",     required_argument,        nullptr,    11},
        {"var1000k",    required_argument,        nullptr,    12},
        {"fix16m",      required_argument,        nullptr,    13},
        {"ignore",      required_argument,        nullptr,    'i'},
        {"help",        no_argument,              nullptr,    'h'},
        {"output",      required_argument,        nullptr,    'o'},
        {"name",        required_argument,        nullptr,    'n'},
        {0, 0, 0, 0}
    };

    int opt_index = 0;
    int c = -1;
    int global_index = 0;
    int samples = 10000;
    int32_t ignore = 10;
    while ((c = getopt_long(argc, argv, "h", long_opts, &opt_index)) != -1) {
        switch (c) {
        case 's':
            samples = atoi(optarg);
            break;
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        {
            int num = atoi(optarg);
            for (int i = 0; i < num; ++i) {
                std::string topic = std::string(long_opts[opt_index].name) + "_" + std::to_string(global_index);
                subs.emplace_back(std::make_shared<SubInfo>(
                        topic,
                        samples,
                        long_opts[opt_index].name,
                        ignore));
                global_index++;

                if ((!file_name.empty()) && (!test_name.empty())) {
                    subs.back()->_file_name = file_name;
                    subs.back()->_test_name = test_name;
                }

                LAT_LOG_INFO << "Build sub topic [" << topic << "] of type [" << std::string(long_opts[opt_index].name) << "]";
            }
            
            break;
        }
        case 'i':
            ignore = atoi(optarg);
            break;

        case 'o':
            file_name = std::string(optarg);
            break;

        case 'n':
            test_name = std::string(optarg);
            break;

        case 'h':
        default:
            Usage();
            break;
        }
    }

    bool end = false;
    while (!end) {
        end = true;
        for (auto& sub_info : subs) {
            if (!sub_info->_stop) {
                end = false;
                break;
            }
        }

        sleep(1);
    }

    LAT_LOG_INFO << "Test finished.";
}
