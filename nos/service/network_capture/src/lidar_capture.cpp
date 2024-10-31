#include "network_capture/include/lidar_capture.h"

#include <stdlib.h>

/*debug*/#include "network_capture/include/function_statistics.h"
#include "network_capture/include/statistics_define.h"
namespace hozon {
namespace netaos {
namespace network_capture {

extern std::uint64_t unprocessed_lidar_frame;
extern std::uint64_t processed_lidar_frame;

static const std::string dead_reckoning_topic = "/perception/parking/slam_location";

std::int32_t LidarCapture::Init() {
    filter_exp = "src host " + lidar_filter_info_.src_host + " and "
                 "dst host " + lidar_filter_info_.dst_host + " and "
                 "src port " + lidar_filter_info_.src_port + " and "
                 "dst port " + lidar_filter_info_.dst_port; 

    send_data_ptr_ = std::make_unique<hozon::soc::RawPointCloud>();
    frame_list_.reserve(800);
    NETWORK_LOG_INFO << "filter_exp : " << filter_exp;

    dead_reckoning_proxy_ = std::make_unique<hozon::netaos::cm::Proxy>(std::make_shared<CmProtoBufPubSubType>());
    latest_dead_reckoning_data_ = std::make_shared<CmProtoBuf>();

    NETWORK_LOG_INFO << "Init cm proxy success. Topic: " << dead_reckoning_topic;
    return true;
}

std::int32_t LidarCapture::Run() {
    NETWORK_LOG_INFO << "LidarCapture::Run()";
    stop_flag_ = false;
    // frame_count = 0;
    if (dead_reckoning_proxy_->Init(0, dead_reckoning_topic) < 0) {
        NETWORK_LOG_ERROR << "Init cm proxy failed. Topic: " << dead_reckoning_topic;
        dead_reckoning_proxy_.reset();
        return false;
    } 
    send_data_ptr_->Clear();
    frame_list_.clear();
    receive_thread_ = std::make_unique<std::thread>(std::thread(&LidarCapture::dead_reckoning_receiver, this));
    lidar_thread_ = std::make_unique<std::thread>(std::thread(&LidarCapture::capPacket, this, lidar_filter_info_.eth_name, filter_exp));
    // frame_ratio_info_thread_ = std::make_unique<std::thread>(std::thread(&LidarCapture::frame_ratio_info, this));
    return true;
}

std::int32_t LidarCapture::Stop() {
    NETWORK_LOG_INFO << "LidarCapture::Stop()";
    stop_flag_ = true;

    if (lidar_thread_->joinable())
        lidar_thread_->join();
    if (receive_thread_->joinable())
        receive_thread_->join();
    dead_reckoning_proxy_->Deinit();
    // if (frame_ratio_info_thread_->joinable())
    //     frame_ratio_info_thread_->join();
    return true;
}

std::int32_t LidarCapture::DeInit() {
    dead_reckoning_proxy_.reset();
    return true;
}

void LidarCapture::capPacket(std::string eth_name, std::string filter_exp) {
    NETWORK_LOG_INFO << "lidar capPacket start...";
    uint count = 0;
    pcap_t *handle;
    char errbuf[PCAP_ERRBUF_SIZE];
    struct pcap_pkthdr *header;
    const u_char *packet;
    struct bpf_program fp;

    // 打开网卡 
    handle = pcap_open_live(eth_name.c_str(), BUFSIZ, 1, 1000, errbuf);
    if (handle == NULL) {
        NETWORK_LOG_ERROR << "Couldn't open device " << eth_name << " : " << errbuf;
        return;
    }
    // 设置过滤条件
    if (pcap_compile(handle, &fp, filter_exp.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {
        NETWORK_LOG_ERROR << "Couldn't parse filter " << filter_exp << " : " << pcap_geterr(handle);
        return;
    }
    if (pcap_setfilter(handle, &fp) == -1) {
        NETWORK_LOG_ERROR << "Couldn't install filter " << filter_exp << " : " << pcap_geterr(handle);
        return;
    }
    if (pcap_set_timeout(handle, 1000) == -1) {
        fprintf(stderr, "Couldn't set timeout: %s\n", pcap_geterr(handle));
        return;
    }
    // 设置非阻塞
    if (pcap_setnonblock(handle, 1, errbuf) == -1) {
        NETWORK_LOG_ERROR << "Error setting non-blocking mode: " << errbuf;
        return;
    }
    // 开始抓包
    while (!stop_flag_) {
        if (0 == pcap_next_ex(handle, &header, &packet)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (stop_flag_) break;
        process_packet(packet, *header);
        count++;
    }

    pcap_close(handle);
    
    NETWORK_LOG_INFO << "capture " << count << " lidar messages...";
    return;
}

void LidarCapture::process_packet(const u_char *packet, pcap_pkthdr& header) {
    // static int msg_count;
    // NETWORK_LOG_ERROR << "msg_count : " << ++msg_count;
    // NETWORK_LOG_INFO << "lidar process_packet";
    if (header.len != header.caplen) {
        NETWORK_LOG_ERROR << "error packet, packet not complete, pacet len: " << header.len << " but capture len: " << header.caplen;
    }
    
    unprocessed_lidar_frame++;
    SubPointCloudFrameDEA *frame = (SubPointCloudFrameDEA *)(packet + 42 + 6);
    frame_list_.emplace_back(*frame);
    //判断头帧
    if (start_frame_flag == true  ) {
        // NETWORK_LOG_INFO << "+++++++++++++++++++++++++++++++++++start_frame ";
        fov_angle = 0;
        pack_num = 0;
        {
            std::lock_guard<std::mutex> lk(proxy_mtx_);
            send_data_ptr_->set_location_data_header_length(latest_dead_reckoning_data_->str().size());
            send_data_ptr_->set_location_data_header(std::string(std::make_move_iterator(latest_dead_reckoning_data_->str().begin()), std::make_move_iterator(latest_dead_reckoning_data_->str().end())));
        }
    }
    start_frame_flag = false;
    
    for (const auto& block : frame->data_body.data_block) {
        int32_t Azimuth = block.azimuth * 256 + block.fine_azimuth;
        if (abs(Azimuth - Azimuth_last) > (50 * 25600)) {
            
            processed_lidar_frame++;
            // NETWORK_LOG_INFO << "success Code_wheel_angle_last is: "<< Azimuth_last / 25600.f;
            // NETWORK_LOG_INFO << "success next frame Code_wheel_angle is: "<< Azimuth / 25600.f;
            framing_flag = true;
        }
        Azimuth_last = Azimuth;
        pack_num++;
        fov_angle += 0.1;
    }

    //判断尾帧
    if (framing_flag == true) {
        start_frame_flag = true;
        {
            std::lock_guard<std::mutex> lk(proxy_mtx_);
            send_data_ptr_->set_location_data_tail_length(latest_dead_reckoning_data_->str().size());
            send_data_ptr_->set_location_data_tail(std::string(std::make_move_iterator(latest_dead_reckoning_data_->str().begin()), std::make_move_iterator(latest_dead_reckoning_data_->str().end())));
        }
        {
            // FunctionStatistics("Get Unix Timestamp ");
            tm_time.tm_year = frame->data_tail.data_time[0];
            tm_time.tm_mon = frame->data_tail.data_time[1] - 1;
            tm_time.tm_mday = frame->data_tail.data_time[2];
            tm_time.tm_hour = frame->data_tail.data_time[3];
            tm_time.tm_min = frame->data_tail.data_time[4];
            tm_time.tm_sec = frame->data_tail.data_time[5];
            tm_time.tm_isdst = 0;
            timestamp = timegm(&tm_time);
        }
        send_data_ptr_->mutable_header()->mutable_sensor_stamp()->set_lidar_stamp(timestamp + frame->data_tail.timestamp * 1.0 / 1000000);
        send_data_ptr_->set_data(reinterpret_cast<const char*>(frame_list_.data()), frame_list_.size() * sizeof(SubPointCloudFrameDEA));
        send_data_ptr_->mutable_header()->set_seq(seq_);
        seq_++;
        // NETWORK_LOG_INFO << "frame fov_angle is: "<<fov_angle;          //打印帧的水平视角范围
        // NETWORK_LOG_INFO << "pointcloud frame pack_num is: " << pack_num;
        // NETWORK_LOG_INFO << "=============================end_frame ";
        {
            std::lock_guard<std::mutex> lk(*mtx_);
            lidar_pub_list_->push(std::move(send_data_ptr_));
        }
        send_data_ptr_ = std::make_unique<hozon::soc::RawPointCloud>();
        frame_list_.clear();
    }
    framing_flag = false;
    return;
}

void LidarCapture::dead_reckoning_receiver() {
    int32_t ret;
    uint32_t count = 0;
    while (!stop_flag_) {
        {
            std::lock_guard<std::mutex> lk(proxy_mtx_);
            ret = dead_reckoning_proxy_->Take(latest_dead_reckoning_data_, 0);
        }
        if (ret < 0) {
            if (++count % 10000 == 0) 
                NETWORK_LOG_WARN << "Take data from cm failed. Topic: " << dead_reckoning_topic << ". fail count : " << count;
        } 
        // else {
        //     NETWORK_LOG_DEBUG << "Take data from cm success!!! Topic: " << dead_reckoning_topic;
        // }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// void LidarCapture::frame_ratio_info() {
//     double ratio;
//     uint16_t count = 0;
//     while (!stop_flag_) {
//         if (count < 20) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             count++;
//             continue;
//         }
//         ratio = frame_count / 10.;
//         if (std::fabs(ratio - 10.0) < 0.5)
//             NETWORK_LOG_INFO << "receive lidar frame num : " << frame_count << ", ratio : " << ratio;
//         else 
//             NETWORK_LOG_WARN << "receive lidar frame num : " << frame_count << ", ratio : " << ratio << ", lidar frame radio abnormal!";
//         frame_count = 0;
//         count = 0;
//     }
// }
}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
