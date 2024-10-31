#include "pub.h"
#include "sub.h"

#include <vector>
#include <memory>
#include <unistd.h>
#include "gen/latPubSubTypes.h"
#include "gen/lat.h"
#include <getopt.h>

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

class PubInfo {
public:
    PubInfo(const std::string& topic, const std::string& type) {
        _type = type;
        publisher = std::make_shared<Publisher>();

        if (type == "fix1k") {
            publisher->Init(topic, std::make_shared<Lat1KPubSubType>());
            _size = 1024;
        }
        else if (type == "fix10k") {
            publisher->Init(topic, std::make_shared<Lat10KPubSubType>());
            _size = 10240;
        }
        else if (type == "fix100k") {
            publisher->Init(topic, std::make_shared<Lat100KPubSubType>());
            _size = 102400;
        }
        else if (type == "fix200k") {
            publisher->Init(topic, std::make_shared<Lat200KPubSubType>());
            _size = 204800;
        }
        else if (type == "fix500k") {
            publisher->Init(topic, std::make_shared<Lat500KPubSubType>());
            _size = 512000;
        }
        else if (type == "fix1000k") {
            publisher->Init(topic, std::make_shared<Lat1000KPubSubType>());
            _size = 1024000;
        }
        else if (_type == "fix16m") {
            publisher->Init(topic, std::make_shared<Lat16mPubSubType>());
            _size = 16777216;
        }
        else if (type == "var1k") {
            publisher->Init(topic, std::make_shared<LatVarPubSubType>());
            _size = 1024;
        }
        else if (type == "var10k") {
            publisher->Init(topic, std::make_shared<LatVarPubSubType>());
            _size = 10240;
        }
        else if (type == "var100k") {
            publisher->Init(topic, std::make_shared<LatVarPubSubType>());
            _size = 102400;
        }
        else if (type == "var200k") {
            publisher->Init(topic, std::make_shared<LatVarPubSubType>());
            _size = 204800;
        }
        else if (type == "var500k") {
            publisher->Init(topic, std::make_shared<LatVarPubSubType>());
            _size = 512000;
        }
        else if (type == "var1000k") {
            publisher->Init(topic, std::make_shared<LatVarPubSubType>());
            _size = 1024000;
        }
        else {
            LAT_LOG_ERROR << "Unknown type " << type; 
        }
    }

    void PubOnce() {
        if (_type.find("var") != std::string::npos) {
            LatVar msg;
            msg.lat().timestamp_us(GetCurrTimeStampUs());
            msg.lat().seq(seq++);
            msg.payload().resize(_size);
            publisher->Write(&msg);
        }
        else {
            if (_type == "fix1k") {
                Lat1K msg;
                msg.lat().timestamp_us(GetCurrTimeStampUs());
                msg.lat().seq(seq++);
                publisher->Write(&msg);
            }
            else if (_type == "fix10k") {
                Lat10K msg;
                msg.lat().timestamp_us(GetCurrTimeStampUs());
                msg.lat().seq(seq++);
                publisher->Write(&msg);
            }
            else if (_type == "fix100k") {
                Lat100K msg;
                msg.lat().timestamp_us(GetCurrTimeStampUs());
                msg.lat().seq(seq++);
                publisher->Write(&msg);
            }
            else if (_type == "fix200k") {
                Lat200K msg;
                msg.lat().timestamp_us(GetCurrTimeStampUs());
                msg.lat().seq(seq++);
                publisher->Write(&msg);
            }
            else if (_type == "fix500k") {
                Lat500K msg;
                msg.lat().timestamp_us(GetCurrTimeStampUs());
                msg.lat().seq(seq++);
                publisher->Write(&msg);
            }
            else if (_type == "fix1000k") {
                Lat1000K msg;
                msg.lat().timestamp_us(GetCurrTimeStampUs());
                msg.lat().seq(seq++);
                publisher->Write(&msg);
            }
            else if (_type == "fix16m") {
                std::shared_ptr<Lat16m> msg(new Lat16m);
                msg->lat().timestamp_us(GetCurrTimeStampUs());
                msg->lat().seq(seq++);
                publisher->Write(msg.get());
            }
            else {
                LAT_LOG_ERROR << "Unknown type " <<_type; 
            }
        }
    }

    std::shared_ptr<Publisher> publisher;
    std::string _type;
    uint32_t _size = 0;
    int32_t seq = 0;
};

int main(int argc, char* argv[]) {
    SetScheduler();

    std::vector<std::shared_ptr<PubInfo>> pubs;

    struct option long_opts[] = {
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
        {"help",        no_argument,              nullptr,    'h'},
        {0, 0, 0, 0}
    };

    int opt_index = 0;
    int c = -1;
    int global_index = 0;
    while ((c = getopt_long(argc, argv, "h", long_opts, &opt_index)) != -1) {
        switch (c) {
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
                pubs.emplace_back(std::make_shared<PubInfo>(
                        topic,
                        long_opts[opt_index].name));
                global_index++;
                LAT_LOG_INFO << "Build pub topic [" << topic << "] of type [" << std::string(long_opts[opt_index].name) << "]";
            }
            break;
        }
        case 'h':
        default:
            Usage();
            break;
        }
    }

    std::chrono::steady_clock::time_point period_wakeup_timepoint = std::chrono::steady_clock::now();
    while (1) {
        std::this_thread::sleep_until(period_wakeup_timepoint + std::chrono::milliseconds(33));
        period_wakeup_timepoint = std::chrono::steady_clock::now();
        for (auto& pub : pubs) {
            pub->PubOnce();
        }
        
    }
}
