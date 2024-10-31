/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef SOMEIP_CAPTURE_H
#define SOMEIP_CAPTURE_H
#pragma once

#include "network_capture/include/base_capture.h"
#include "network_capture/include/someip_capture_config.h"
#include "network_capture/include/someip_struct_define.h"
#include "network_capture/include/network_publisher.h"

namespace hozon {
namespace netaos {
namespace network_capture {

class SomeipCapture : protected BaseCapture {
   public:
    std::int32_t Init() override;
    std::int32_t DeInit() override;
    std::int32_t Run() override;
    std::int32_t Stop() override;

   protected:
    void capPacket(std::string eth_name, std::string filter_exp) override;
    void process_packet(const u_char *packet, pcap_pkthdr& header) override;

   private:
    bool someip_tp_write(const char* p_payload, const std::unique_ptr<someip_message_t>& someip_msg, const pcap_pkthdr& header, const std::string& topic_name);
    void someip_write(const char* p_payload, const std::unique_ptr<someip_message_t>& someip_msg, const pcap_pkthdr& header, const std::string& topic_name);
    void someip_header_to_message_header(const someip_hdr_t &p_someip_hdr, std::unique_ptr<someip_message_t> &someip_msg);
    void someip_ratio_info();
    void parse_udp(const u_char *packet, pcap_pkthdr& header);
    void parse_tcp(const u_char *packet, pcap_pkthdr& header);

   private:
    bool stop_flag_ = false;
    std::map<std::string, std::vector<std::string>> topics_map_;
    std::map<uint32_t, std::vector<someip_tp_segment_t>> someip_tp_map_;
    SomeipFilterInfo someip_filter_info_;
    std::unique_ptr<std::thread> someip_thread_;
    std::string filter_exp;

    std::unique_ptr<raw_someip_message> send_data_ptr_;
    std::shared_ptr<std::queue<std::unique_ptr<raw_someip_message>>> someip_pub_list_;
    std::shared_ptr<std::mutex> mtx_;

    // std::unique_ptr<std::thread> someip_ratio_info_thread_;
    // uint64_t frame_count = 0;

   public:
    SomeipCapture(const SomeipFilterInfo &info, const std::shared_ptr<std::queue<std::unique_ptr<raw_someip_message>>> &someip_pub_list, const std::shared_ptr<std::mutex> &mtx) 
    : someip_filter_info_(info)
    , someip_pub_list_(someip_pub_list)
    , mtx_(mtx) {}
};

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif