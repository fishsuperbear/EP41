#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include "ara/com/sample_ptr.h"
#include "hozon/netaos/impl_type_ussrawdataset.h"
#include "common.h"
#include "proto/soc/uss_rawdata.pb.h"
#include "logger.h"
#include "param_config.h"

namespace hozon {
namespace netaos {
namespace sensor {
#define SENSOR_USSECHO_TRANS(ussEcho, data) \
        {   \
            proto_data->mutable_##ussEcho()->set_counter(data.counter);  \
            proto_data->mutable_##ussEcho()->set_status_work(data.status_work);  \
            proto_data->mutable_##ussEcho()->set_status_error(data.status_error);   \
            proto_data->mutable_##ussEcho()->set_system_time(                         \
                        static_cast<double>(data.system_time >> 32u)                   \
                        + static_cast<double>((data.system_time & 0xffffffff) / 1e9)); \
            proto_data->mutable_##ussEcho()->set_wtxsns_ringtime(data.wTxSns_Ringtime); \
            proto_data->mutable_##ussEcho()->set_reserveda(data.ReservedA);    \
            proto_data->mutable_##ussEcho()->set_reservedb(data.ReservedB);  \
            for (auto it : data.distance) {  \
                proto_data->mutable_##ussEcho()->add_distance(it);  \
            }   \
            for (auto it : data.width) {   \
                proto_data->mutable_##ussEcho()->add_width(it);   \
            }    \
            for(auto it : data.peak) {    \
                proto_data->mutable_##ussEcho()->add_peak(it);    \
            }       \
            proto_data->mutable_##ussEcho()->set_echo_num(data.echo_num);   \
                    \
        } 
class UssProxy {
public:
    UssProxy() : _uss_pub_last_seq(0) {}
    ~UssProxy() = default;
    std::shared_ptr<hozon::soc::UssRawDataSet> Trans(ara::com::SamplePtr<::hozon::netaos::UssRawDataSet const> data) {

        std::shared_ptr<hozon::soc::UssRawDataSet> proto_data = std::make_shared<hozon::soc::UssRawDataSet>();
        if (proto_data == nullptr) {
            SENSOR_LOG_ERROR << "ussRawData_proto Allocate got nullptr!";
            return nullptr;
        }
        proto_data->mutable_header()->set_seq(GetCount());
        proto_data->mutable_header()->set_frame_id("ussRawData");;
        proto_data->mutable_header()->set_publish_stamp(GetRealTimestamp());
        proto_data->mutable_header()->mutable_sensor_stamp()->set_uss_stamp(
            static_cast<double>(data->time_stamp >> 32u) 
            + static_cast<double>((data->time_stamp & 0xffffffff) / 1e9));
        proto_data->mutable_header()->set_gnss_stamp(GetAbslTimestamp());


        proto_data->set_counter(data->counter);
    
        SENSOR_USSECHO_TRANS(flc_info, data->flc_info); 
        SENSOR_USSECHO_TRANS(frs_info, data->frs_info);
        SENSOR_USSECHO_TRANS(rls_info, data->rls_info);
        SENSOR_USSECHO_TRANS(rrs_info, data->rrs_info);
        SENSOR_USSECHO_TRANS(rrc_info, data->rrc_info);
        SENSOR_USSECHO_TRANS(flm_info, data->flm_info);
        SENSOR_USSECHO_TRANS(frm_info, data->frm_info);
        SENSOR_USSECHO_TRANS(frc_info, data->frc_info);
        SENSOR_USSECHO_TRANS(rlm_info, data->rlm_info);
        SENSOR_USSECHO_TRANS(rrm_info, data->rrm_info);
        SENSOR_USSECHO_TRANS(fls_info, data->fls_info);
        SENSOR_USSECHO_TRANS(rlc_info, data->rlc_info);
        
        check_uss_status("uss_flc", data->flc_info.status_work);
        check_uss_status("uss_frs", data->frs_info.status_work);
        check_uss_status("uss_rls", data->rls_info.status_work);
        check_uss_status("uss_rrs", data->rrs_info.status_work);
        check_uss_status("uss_rrc", data->rrc_info.status_work);
        check_uss_status("uss_flm", data->flm_info.status_work);
        check_uss_status("uss_frm", data->frm_info.status_work);
        check_uss_status("uss_frc", data->frc_info.status_work);
        check_uss_status("uss_rlm", data->rlm_info.status_work);
        check_uss_status("uss_rrm", data->rrm_info.status_work);
        check_uss_status("uss_fls", data->fls_info.status_work);
        check_uss_status("uss_rlc", data->rlc_info.status_work);

        if(_uss_pub_last_seq && (proto_data->mutable_header()->seq() > _uss_pub_last_seq) 
            && ((proto_data->mutable_header()->seq() - _uss_pub_last_seq) != 1)) {
            SENSOR_LOG_WARN << "chassis info loss data. receive seq: " << proto_data->mutable_header()->seq() \
                << " last seq : "  << _uss_pub_last_seq  \
                << " seq diff : " << (proto_data->mutable_header()->seq() - _uss_pub_last_seq)
                << " interval : " << (proto_data->mutable_header()->publish_stamp() \
                    - proto_data->mutable_header()->mutable_sensor_stamp()->uss_stamp()) << " s";
        } else if ((proto_data->mutable_header()->publish_stamp() 
                - proto_data->mutable_header()->mutable_sensor_stamp()->uss_stamp()) > 0.02f) {  // 20 ms
            SENSOR_LOG_WARN << "chassis info link latency : " << (proto_data->mutable_header()->publish_stamp() \
                    - proto_data->mutable_header()->mutable_sensor_stamp()->uss_stamp()) << " s";
        }
        _uss_pub_last_seq = proto_data->mutable_header()->seq();

        if (!(proto_data->mutable_header()->seq() % 100)) {
            PRINTSENSORDATA(proto_data->mutable_header()->seq());
            PrintOriginalData(data);
            struct timespec time;
            if (0 != clock_gettime(CLOCK_REALTIME, &time)) {
                SENSOR_LOG_WARN << "clock_gettime fail ";
            }
            
            uint64_t current_time = (uint64_t)time.tv_sec * 1000lu + ((uint64_t)time.tv_nsec)/1000lu/1000lu;
            uint64_t time_diff = 0;
            if(uss_pub_last_time && (((time_diff = current_time - uss_pub_last_time) - 2000lu) > 200lu)) {
                SENSOR_LOG_WARN << "Send uss info counter: " << proto_data->mutable_header()->seq() \
                    << " ,interval : " << time_diff << " ms";
            }
            else {
                SENSOR_LOG_INFO << "Send uss info counter: " << proto_data->mutable_header()->seq() \
                    << " ,interval : " << time_diff << " ms";
            }
            uss_pub_last_time = current_time;
        }
        return proto_data;
    }
private:
    int PrintOriginalData(ara::com::SamplePtr<::hozon::netaos::UssRawDataSet const> data) {
        PRINTSENSORDATA(data->counter);
        PRINTSENSORDATA(data->time_stamp);
        PRINTSENSORDATA(data->flc_info.counter);
        PRINTSENSORDATA(data->flc_info.system_time);
        PRINTSENSORDATA(data->rrc_info.ReservedB);
        PRINTSENSORDATA(data->flc_info.status_work);
        return 0;
    }
    int check_uss_status(std::string name, uint8_t status) {
        uint8_t work_status = 0;
        switch(status) {
            case 0:
                work_status = 2;   // no work
                break;
            case 1:
            case 2:
            case 3:
                work_status = 1;  // work
                break;
        }
        std::string key = "system/" + name + "_status";
        if(_uss_status_map.find(name) != _uss_status_map.end()) {
            if(_uss_status_map[name] != work_status) {
                if(ParamConfig::GetInstance().SetParam(key, work_status) == 0) {
                    _uss_status_map[name] = work_status;
                    SENSOR_LOG_INFO << "Set " << key << " to " << work_status << " success.";
                    return 0;
                }
            }
        }
        else {
            if(ParamConfig::GetInstance().SetParam(key, work_status) == 0) {
                _uss_status_map[name] = work_status;
                SENSOR_LOG_INFO << "Set " << key << " to " << work_status << " success.";
                return 0;
            }   
        }
        // NODE_LOG_INFO << "Set " << key << " to " << work_status << " fail.";
        return -1;
    }
    int32_t GetCount() {
        static int32_t count = 0;
        if(count++ < 0) {
            count = 0;
        }
        return count;
    }
    uint64_t uss_pub_last_time;
    uint32_t _uss_pub_last_seq;
    std::unordered_map<std::string, uint8_t> _uss_status_map;
};


}
}
}
