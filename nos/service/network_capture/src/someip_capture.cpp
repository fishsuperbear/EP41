#include "network_capture/include/someip_capture.h"
#include "network_capture/include/statistics_define.h"

namespace hozon {
namespace netaos {
namespace network_capture {

extern std::atomic<std::uint64_t> unprocessed_someip_frame;
extern std::atomic<std::uint64_t> udp_frame;
extern std::atomic<std::uint64_t> tcp_frame;
extern std::atomic<std::uint64_t> processed_someip_frame;
extern std::atomic<std::uint64_t> someip_frame;
// static std::atomic<std::uint64_t> processed_someip_frame = 0;
extern std::atomic<std::uint64_t> someip_tp_frame;


std::int32_t SomeipCapture::Init() {
    for (size_t i = 0; i < someip_filter_info_.ports.size(); ++i) {
        filter_exp += (i ? " or port " : "port ") + std::to_string(someip_filter_info_.ports[i]);  
    }

    send_data_ptr_ = std::make_unique<raw_someip_message>();

    NETWORK_LOG_INFO << "filter_exp : " << filter_exp << ", eth_name : " << someip_filter_info_.eth_name;
    return true;
}
std::int32_t SomeipCapture::Run() {
    // 创建someip抓包线程
    stop_flag_ = false;
    // frame_count = 0;
    someip_thread_ = std::make_unique<std::thread>(std::thread(&SomeipCapture::capPacket, this, someip_filter_info_.eth_name, filter_exp));
    // someip_ratio_info_thread_ = std::make_unique<std::thread>(std::thread(&SomeipCapture::someip_ratio_info, this));
    return true;
}
std::int32_t SomeipCapture::Stop() {
    stop_flag_ = true;
    if (someip_thread_->joinable())
        someip_thread_->join();
    // if (someip_ratio_info_thread_->joinable())
    //     someip_ratio_info_thread_->join();
    return true;
}
std::int32_t SomeipCapture::DeInit() {
    topics_map_.clear();
    someip_tp_map_.clear();
    return true;
}

void SomeipCapture::capPacket(std::string eth_name, std::string filter_exp) {
    NETWORK_LOG_INFO << "SomeipCapture capPacket start, capture at : " << eth_name;
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
    
    NETWORK_LOG_INFO << "capture " << count << " someip messages...";
    return;
}

void SomeipCapture::process_packet(const u_char *packet, pcap_pkthdr& header) {
    debug_printf("------------------------------------------------------------------------------------------------------\n");
    if (header.len != header.caplen) {
        NETWORK_LOG_ERROR << "error packet, packet not complete, pacet len: " << header.len << " but capture len: " << header.caplen;
    } 

    // 检查是否是IPv4数据包
    if (*((u_int16_t*) (packet + 12)) == htons(0x0800)) {
        
        Counter::Instance().increment(unprocessed_someip_frame);
        struct ip *iph = (struct ip *)(packet + 14);
        uint8_t tcp_or_udp = iph->ip_p;
        if (tcp_or_udp == IPPROTO_UDP) {
            
            Counter::Instance().increment(udp_frame);
            parse_udp(packet, header);
        } else if (tcp_or_udp == IPPROTO_TCP) {
            
            Counter::Instance().increment(tcp_frame);
            parse_tcp(packet, header);
        } else {
            NETWORK_LOG_ERROR << "error packet, ip proto: " << iph->ip_p;
            return;
        }
    }
    
    return;
}

void SomeipCapture::parse_tcp(const u_char *packet, pcap_pkthdr& header) {
    const u_char* pkt_data = packet;
    uint32_t offset = 0;
    uint32_t remain_len = 0;
    uint32_t tcp_headlen = 0;
    // 检查是否是IPv4数据包
    if (*((u_int16_t*) (pkt_data + 12)) == htons(0x0800)) {
        struct ip *iph = (struct ip *)(pkt_data + 14);
        uint8_t tcp_or_udp = iph->ip_p;
        remain_len = ntohs(iph->ip_len);

        // 跳过Ethernet头部（通常为14字节）
        pkt_data += 14;
        offset += 14;

        // 获取IP头部长度
        int ip_header_length = (int) (*(pkt_data + 0) & 0x0F) * 4;
        
        // 跳过IP头部
        pkt_data += ip_header_length;
        offset += ip_header_length;
        remain_len -= ip_header_length;

        if (tcp_or_udp == IPPROTO_TCP) {
            // 获取TCP头部长度
            tcp_headlen = ((*(pkt_data + 12) >> 4) & 0x0F) * 4;
            offset += tcp_headlen;
            remain_len -= tcp_headlen;
        } else {
            return;
        }
    } else {
        return;
    }

    if (remain_len < SOMEIP_FULL_HDR_SIZE) {
        return;
    }

    const u_char *tcp_payload = packet + offset;
    someip_hdr_t *p_someip_hdr = (someip_hdr_t *)(tcp_payload);
    if (p_someip_hdr->msg_type == 0x00         || ntohs(p_someip_hdr->message_id.service_id) == 0xffff || 
        p_someip_hdr->protocol_version != 0x01 || p_someip_hdr->interface_version != 0x01) {
        return;
    }
    std::string topic_name;

    std::unique_ptr<someip_message_t> someip_msg = std::make_unique<someip_message_t>();
    someip_header_to_message_header(*p_someip_hdr, someip_msg);
    uint32_t message_id = MessageID(someip_msg->someip_hdr.message_id.service_id, someip_msg->someip_hdr.message_id.method_id);
    topic_name = someip_filter_info_.topic_map[message_id];
    debug_printf("message_id : %10x topic_name : %s\n", message_id, topic_name.c_str());
    char *p_payload = (char *)(tcp_payload + SOMEIP_FULL_HDR_SIZE);
    someip_write(p_payload, someip_msg, header, topic_name);
    return;
}

void SomeipCapture::parse_udp(const u_char *packet, pcap_pkthdr& header) {
    struct ip *iph = (struct ip *)(packet + 14);
    uint32_t offset = 0;
    uint16_t remain_len = ntohs(iph->ip_len);
    // debug_printf("** remain_len : %x\n", remain_len);
    if (iph->ip_p == IPPROTO_UDP) {
        remain_len -= 20;
        offset = 42; // 14 + 20 + 8
    } else {
        NETWORK_LOG_ERROR << "error packet, ip proto: " << iph->ip_p;
        return;
    }
    while (remain_len >= 8 && offset != ntohs(iph->ip_len) + 14u && !stop_flag_) {
        // debug_printf("remain_len : %4u, offset %3u\n", remain_len, offset);
        someip_hdr_t *p_someip_hdr = (someip_hdr_t *)(packet + offset);
        if (p_someip_hdr->msg_type == 0x00         || ntohs(p_someip_hdr->message_id.service_id) == 0xffff || 
            p_someip_hdr->protocol_version != 0x01 || p_someip_hdr->interface_version != 0x01) 
            return;
        std::string topic_name;

        std::unique_ptr<someip_message_t> someip_msg = std::make_unique<someip_message_t>();
        // if (ntohs(p_someip_hdr->message_id.service_id) != 0x61b6)
        //         return;
        // if (ntohs(p_someip_hdr->message_id.method_id) != 0x8012)
        //         return;
        someip_header_to_message_header(*p_someip_hdr, someip_msg);
        uint32_t message_id = MessageID(someip_msg->someip_hdr.message_id.service_id, someip_msg->someip_hdr.message_id.method_id);
        topic_name = someip_filter_info_.topic_map[message_id];
        debug_printf("message_id : %10x topic_name : %s\n", message_id, topic_name.c_str());
        char *p_payload = (char *)(packet + offset + SOMEIP_FULL_HDR_SIZE);
        if (p_someip_hdr->msg_type & 0b00100000) {
            if(someip_tp_write(p_payload, someip_msg, header, topic_name)) break;
        } else {
            someip_write(p_payload, someip_msg, header, topic_name);
        }

        remain_len -= (someip_msg->someip_hdr.length + 8);
        offset += (someip_msg->someip_hdr.length + 8);
    }
    debug_printf("%s\n", std::string(100, '-').c_str());
    return;
}

// void SomeipCapture::process_packet(const u_char *packet, pcap_pkthdr& header) {
//     debug_printf("------------------------------------------------------------------------------------------------------\n");
//     if (header.len != header.caplen) {
//         NETWORK_LOG_ERROR << "error packet, packet not complete, pacet len: " << header.len << " but capture len: " << header.caplen;
//     }
//     // struct ip *iph = (struct ip *)(packet + 14);
//     // uint32_t offset = 0;
//     // uint16_t remain_len = ntohs(iph->ip_len);
//     // // debug_printf("** remain_len : %x\n", remain_len);
//     // if (iph->ip_p == IPPROTO_UDP) {
//     //     remain_len -= 20;
//     //     offset = 42; // 14 + 20 + 8
//     // } else if (iph->ip_p == IPPROTO_TCP) {
//     //     remain_len -= 40;
//     //     offset = 54; // 14 + 20 + 20
//     // } else {
//     //     NETWORK_LOG_ERROR << "error packet, ip proto: " << iph->ip_p;
//     //     return;
//     // }


//     const u_char* pkt_data = packet;
//     uint32_t offset = 0;
//     uint16_t remain_len = 0;
//     // 检查是否是IPv4数据包
//     if (*((u_int16_t*) (pkt_data + 12)) == htons(0x0800)) {
//         struct ip *iph = (struct ip *)(pkt_data + 14);
//         uint8_t tcp_or_udp = iph->ip_p;
//         remain_len = ntohs(iph->ip_len);

//         // 跳过Ethernet头部（通常为14字节）
//         pkt_data += 14;
//         offset += 14;
//         remain_len -= 14;

//         // 获取IP头部长度
//         int ip_header_length = (int) (*(pkt_data + 0) & 0x0F) * 4;
        
//         // 跳过IP头部
//         pkt_data += ip_header_length;
//         offset += ip_header_length;
        
//         if (tcp_or_udp == IPPROTO_UDP) {
//             offset += 8; // 14 + 20 + 8
//         } else if (tcp_or_udp == IPPROTO_TCP) {
//             // 获取TCP头部长度
//             offset += ((*(pkt_data + 12) >> 4) & 0x0F) * 4;
//         } else {
//             NETWORK_LOG_ERROR << "error packet, ip proto: " << iph->ip_p;
//             return;
//         }
//     }


//     while (remain_len >= 8 && !stop_flag_) {
//         // debug_printf("remain_len : %4u, offset %3u\n", remain_len, offset);
//         someip_hdr_t *p_someip_hdr = (someip_hdr_t *)(packet + offset);
//         if (p_someip_hdr->msg_type == 0x00         || ntohs(p_someip_hdr->message_id.service_id) == 0xffff || 
//             p_someip_hdr->protocol_version != 0x01 || p_someip_hdr->interface_version != 0x01) 
//             return;
//         std::string topic_name;

//         std::unique_ptr<someip_message_t> someip_msg = std::make_unique<someip_message_t>();
//         // if (ntohs(p_someip_hdr->message_id.service_id) != 0x61b6)
//         //         return;
//         // if (ntohs(p_someip_hdr->message_id.method_id) != 0x8012)
//         //         return;
//         someip_header_to_message_header(*p_someip_hdr, someip_msg);
//         uint32_t message_id = MessageID(someip_msg->someip_hdr.message_id.service_id, someip_msg->someip_hdr.message_id.method_id);
//         topic_name = someip_filter_info_.topic_map[message_id];
//         debug_printf("message_id : %10x topic_name : %s\n", message_id, topic_name.c_str());
//         char *p_payload = (char *)(packet + offset + SOMEIP_FULL_HDR_SIZE);
//         if (p_someip_hdr->msg_type & 0b00100000) {
//             if(someip_tp_write(p_payload, someip_msg, header, topic_name)) break;
//         } else {
//             someip_write(p_payload, someip_msg, header, topic_name);
//         }

//         remain_len -= (someip_msg->someip_hdr.length + 8);
//         offset += (someip_msg->someip_hdr.length + 8);
//     }
//     debug_printf("%s\n", std::string(100, '-').c_str());
//     return;
// }

bool SomeipCapture::someip_tp_write(const char* p_payload, const std::unique_ptr<someip_message_t>& someip_msg, const pcap_pkthdr& header, const std::string& topic_name) {
    uint32_t someip_tp_header = ntohl(*(uint32_t*)p_payload);
    uint32_t offset_tp_payload = ((someip_tp_header >> 4) & 0xFFFFFFF) << 4; // offset_tp_payload 单位为 16bit
    uint32_t msg_id = MessageID(someip_msg->someip_hdr.message_id.service_id, someip_msg->someip_hdr.message_id.method_id);
    bool is_more = someip_tp_header & 0b1;
    bool is_segment_exist = false, is_lack = false; 
    debug_printf("someip-tp header: %08x\n", someip_tp_header);
    debug_printf("offset_tp_payload: %d, is_more: %d\n", offset_tp_payload, is_more);
    someip_tp_segment_t someip_tp_msg;
    someip_tp_msg.offset = offset_tp_payload;
    someip_tp_msg.is_more = is_more;
    someip_tp_msg.length = someip_msg->data_len - 4;
    someip_tp_msg.payload = new char[someip_msg->data_len - 4];
    memcpy(someip_tp_msg.payload, p_payload + 4, someip_msg->data_len - 4);
    for (size_t i = 0; i < someip_tp_map_[msg_id].size(); ++i) {
        if (someip_tp_msg.offset == someip_tp_map_[msg_id][i].offset) {
            is_segment_exist = true;
            delete[] someip_tp_map_[msg_id][i].payload;
            someip_tp_map_[msg_id][i] = someip_tp_msg;
            break;
        }
    }
    if (!is_segment_exist) {
        someip_tp_map_[msg_id].emplace_back(someip_tp_msg);
    }
    std::sort(someip_tp_map_[msg_id].begin(), someip_tp_map_[msg_id].end(), [](const someip_tp_segment_t& seg1, const someip_tp_segment_t& seg2) {
        if (seg1.offset < seg2.offset) {
            return true;
        } else if (seg1.offset == seg2.offset) {
            return seg1.is_more < seg2.is_more;
        } else {
            return false;
        }
    });
    for (size_t i = 0; i < someip_tp_map_[msg_id].size(); ++i) {
        debug_printf("    %2lu : offset_tp_payload: %d, is_more: %d length: %d\n", i, someip_tp_map_[msg_id][i].offset, someip_tp_map_[msg_id][i].is_more, someip_tp_map_[msg_id][i].length);
    }
    if (false == someip_tp_map_[msg_id][someip_tp_map_[msg_id].size() - 1].is_more) {
        for (size_t i = 0; i < someip_tp_map_[msg_id].size() - 1; ++i) {
            if (someip_tp_map_[msg_id][i].offset / someip_tp_map_[msg_id][0].length != i){
                is_lack = true;
                break;
            }          
        }
        if (!is_lack) {
            uint32_t merge_data_len = 0;
            for (size_t i = 0; i < someip_tp_map_[msg_id].size(); ++i) {
                merge_data_len += someip_tp_map_[msg_id][i].length;
            }
            someip_msg->data_len = merge_data_len;
            someip_msg->someip_hdr.length = merge_data_len + 8;
            someip_msg->someip_hdr.msg_type &= 0xdf;
            debug_printf("someip_msg->data_len : %d\n", someip_msg->data_len);
            debug_printf("someip_msg->someip_hdr.length : %d\n", someip_msg->someip_hdr.length);
            send_data_ptr_->topic = topic_name;
            send_data_ptr_->msg.insert(send_data_ptr_->msg.end(), reinterpret_cast<char*>(someip_msg.get()), reinterpret_cast<char*>(someip_msg.get()) + sizeof(someip_message_t));
            // std::vector<char> someip_message_t_(reinterpret_cast<char*>(someip_msg.get()), reinterpret_cast<char*>(someip_msg.get()) + sizeof(someip_message_t));
            for (size_t i = 0; i < someip_tp_map_[msg_id].size(); ++i) {
                send_data_ptr_->msg.insert(send_data_ptr_->msg.end(), someip_tp_map_[msg_id][i].payload, someip_tp_map_[msg_id][i].payload + someip_tp_map_[msg_id][i].length);
                delete[] someip_tp_map_[msg_id][i].payload;
            }
            {
                std::lock_guard<std::mutex> lk(*mtx_);
                someip_pub_list_->push(std::move(send_data_ptr_));
            }
            send_data_ptr_ = std::make_unique<raw_someip_message>();
            someip_tp_map_[msg_id].clear();
            Counter::Instance().increment(processed_someip_frame);
            Counter::Instance().increment(someip_frame);
            Counter::Instance().increment(std::move(topic_name));
            // uint64_t time_stamp = static_cast<int64_t>(header.ts.tv_sec) * 1000000 + header.ts.tv_usec;
            // Write(someip_message_t_, topic_name, "someip_message_t", time_stamp);
            
            return true;
        }
    }
    return false;
}

void SomeipCapture::someip_write(const char* p_payload, const std::unique_ptr<someip_message_t>& someip_msg, const pcap_pkthdr& header, const std::string& topic_name) {
    send_data_ptr_->topic = topic_name;
    send_data_ptr_->msg.insert(send_data_ptr_->msg.end(), reinterpret_cast<char*>(someip_msg.get()), reinterpret_cast<char*>(someip_msg.get()) + sizeof(someip_message_t));
    // std::vector<char> someip_message_t_(reinterpret_cast<char*>(someip_msg.get()), reinterpret_cast<char*>(someip_msg.get()) + sizeof(someip_message_t));
    send_data_ptr_->msg.insert(send_data_ptr_->msg.end(), p_payload, p_payload + someip_msg->data_len);
    {
        std::lock_guard<std::mutex> lk(*mtx_);
        someip_pub_list_->push(std::move(send_data_ptr_));
    }
    send_data_ptr_ = std::make_unique<raw_someip_message>();
    Counter::Instance().increment(processed_someip_frame);
    Counter::Instance().increment(someip_tp_frame);
    Counter::Instance().increment(std::move(topic_name));
    // uint64_t time_stamp = static_cast<int64_t>(header.ts.tv_sec) * 1000000 + header.ts.tv_usec;
    // Write(someip_message_t_, topic_name, "someip_message_t", time_stamp);
}

void SomeipCapture::someip_header_to_message_header(const someip_hdr_t &p_someip_hdr, std::unique_ptr<someip_message_t> &someip_msg) {
    someip_msg->someip_hdr.message_id.service_id = ntohs(p_someip_hdr.message_id.service_id);
    someip_msg->someip_hdr.message_id.method_id = ntohs(p_someip_hdr.message_id.method_id);
    someip_msg->someip_hdr.length = ntohl(p_someip_hdr.length);
    someip_msg->someip_hdr.request_id.client_id = ntohs(p_someip_hdr.request_id.client_id);
    someip_msg->someip_hdr.request_id.session_id = ntohs(p_someip_hdr.request_id.session_id);
    someip_msg->someip_hdr.protocol_version = p_someip_hdr.protocol_version;
    someip_msg->someip_hdr.interface_version = p_someip_hdr.interface_version;
    someip_msg->someip_hdr.msg_type = p_someip_hdr.msg_type;
    someip_msg->someip_hdr.return_code = p_someip_hdr.return_code;
    someip_msg->data_len = someip_msg->someip_hdr.length - 8;

    // DEBUG 
    debug_printf("service_id: 0x%04x", ntohs(p_someip_hdr.message_id.service_id));
    debug_printf(" method_id: 0x%04x", ntohs(p_someip_hdr.message_id.method_id));
    debug_printf(" length: %4d", ntohl(p_someip_hdr.length));
    debug_printf(" client_id: 0x%04x", ntohs(p_someip_hdr.request_id.client_id));
    debug_printf(" session_id: 0x%04x", ntohs(p_someip_hdr.request_id.session_id));
    debug_printf(" protocol_version: 0x%02x", p_someip_hdr.protocol_version);
    debug_printf(" interface_version: 0x%02x", p_someip_hdr.interface_version);
    debug_printf(" msg_type: 0x%02x", p_someip_hdr.msg_type);
    debug_printf(" return_code: 0x%02x \n", p_someip_hdr.return_code);
}

// void SomeipCapture::someip_ratio_info() {
//     uint16_t count = 0;
//     while (!stop_flag_) {
//         if (count < 20) {
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));
//             count++;
//             continue;
//         }
//         NETWORK_LOG_INFO << "receive all someip frame num : " << frame_count << ", all someip ratio : " << frame_count / 10.;
//         frame_count = 0;
//         count = 0;
//     }
// }

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
