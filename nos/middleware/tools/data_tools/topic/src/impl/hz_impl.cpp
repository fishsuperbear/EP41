
#include "impl/hz_impl.h"
#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <fstream>
#include <unordered_map>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/attributes/SubscriberAttributes.h>
#include <fastrtps/rtps/common/Types.h>
#include <fastrtps/subscriber/SampleInfo.h>
#include <fastrtps/types/DynamicDataFactory.h>
#include <fastrtps/types/DynamicDataHelper.hpp>
#include <fastrtps/types/DynamicPubSubType.h>
#include <fastrtps/types/DynamicTypePtr.h>
#include <fastrtps/types/TypeObjectFactory.h>
#include <google/protobuf/compiler/parser.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include "adf_lite_utile.h"
#include "cm/include/cm_config.h"
#include "data_tools_logger.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"
#include "idl/generated/zerocopy_imageTypeObject.h"
#include "proto_factory.h"
#include "process_utility.h"
#include "topic_manager.hpp"
#include "proto/test/soc/dbg_msg.pb.h"

#define UNUSED_VAR(var) void var

namespace hozon {

namespace netaos {
namespace topic {

HzImpl::HzImpl() {}

HzImpl::~HzImpl() {
    if (!_isStop) {
        Stop();
    }
}

void HzImpl::Stop() {

    _isStop = true;
    // if (hzinfo_thread_.joinable()) {
    //     hzinfo_thread_.join();
    // }

    // SubBase::Stop();
}

void HzImpl::Start(HzOptions hz_options) {

    hz_options_ = hz_options;
    _monitor_all = hz_options.monitor_all;
    _method = hz_options.method;

    if (hz_options_.exit_time > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(hz_options_.exit_time));
        Stop();
    }

    TOPIC_LOG_DEBUG << "topic hz start.";
    SubBase::Start(hz_options.events);

    hzinfo_thread_ = std::thread(&HzImpl::OutputHzInfos, this);

    if (hz_options_.exit_time > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(hz_options_.exit_time));
    } else {
        while (!_isStop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    if (hzinfo_thread_.joinable()) {
        hzinfo_thread_.join();
    }

    SubBase::Stop();

    return;
}

void HzImpl::OutputHzInfos() {
    if (!hz_options_.method) {
        std::cout << "\033[33m"
                  << "method topics won't be showed. Can use -m to show them."
                  << "\033[0m" << std::endl;
    }

    _last_print_time = std::chrono::steady_clock::now();
    while (!_isStop) {
        {
            std::lock_guard<std::mutex> lk(topic_infos_mutex_);
            //delete reader
            for (auto it = topics_.begin(); it != topics_.end();) {
                if (!it->first->is_enabled()) {
                    int32_t dds_data_type = topic_manager_->GetTopicDataType(it->second->get_name());
                    auto subscriber = topic_manager_->GetSubscriber(dds_data_type);
                    subscriber->delete_datareader(it->first);
                    Topic* topic_ptr = it->second;
                    it = topics_.erase(it);
                    _subTopics.erase(topic_ptr->get_name());
                } else {
                    it++;
                }
            }

            const int32_t d_precision = 3;
            const int32_t hz_width = 10;
            int32_t topic_name_max_len = 57;
            for (auto item : topic_infos_) {
                if (item.first.size() > topic_name_max_len) {
                    topic_name_max_len = item.first.size();
                }
            }
            int32_t topic_col_width = topic_name_max_len + 2;

            std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
            // 获取当前系统时间
            auto systemTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            // 将时间转换为本地时间结构
            struct std::tm* timeInfo = std::localtime(&systemTime);
            std::ostringstream oss;
            oss << std::put_time(timeInfo, "%T");
            std::string time_str = "topic - " + oss.str();
            std::cout << std::left << std::setw(topic_col_width) << time_str << std::fixed << std::setprecision(d_precision) << std::setw(hz_width) << "rate" << std::setw(hz_width) << "max_delta"
                      << std::setw(hz_width) << "win_size" << std::setw(hz_width) << "state" << std::endl;

            for (auto item : topic_infos_) {
                if (item.first.size() > topic_name_max_len) {
                    topic_name_max_len = item.first.size();
                }
                if (std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - item.second.last_time).count() > 5) {
                    //超过2s没有收到消息，重置windows
                    topic_infos_[item.first].count = 0;
                    topic_infos_[item.first].min_delta = std::chrono::duration<double>(10.0);
                    topic_infos_[item.first].max_delta = std::chrono::duration<double>(0);
                    item.second.count = 0;
                }

                if (!item.second.is_matched) {
                    //topic不存在，显示未连接
                    std::cout << "\033[31m" << std::left << std::setw(topic_col_width) << item.first << std::fixed << std::setprecision(d_precision) << std::setw(hz_width) << 0 << std::setw(hz_width) << 0
                              << std::setw(hz_width) << 0 << std::setw(hz_width) << "disconnected"
                              << "\033[0m" << std::endl;
                } else if (0 == item.first.find("/lite/") && !hozon::netaos::data_tool_common::IsLiteInfoTopic(item.first)) {
                    if (item.second.count > 0) {
                        std::cout << std::left << std::setw(topic_col_width) << item.first << std::fixed << std::setprecision(d_precision) << std::setw(hz_width) << item.second.alit_fre
                            << std::setw(hz_width) << item.second.max_delta.count() << std::setw(hz_width) << item.second.count << std::setw(hz_width) << "normal" << std::endl;
                    } else {
                        std::cout << std::left << std::setw(topic_col_width) << item.first << std::fixed << std::setprecision(d_precision) << std::setw(hz_width) << 0  << std::setw(hz_width) << 0 << std::setw(hz_width)
                                  << 0 << std::setw(hz_width) << "no message" << std::endl;
                    }

                } else if (item.second.count > hz_options_.skip_sample_num + 1) {
                    std::cout << std::left << std::setw(topic_col_width) << item.first << std::fixed << std::setprecision(d_precision) << std::setw(hz_width)
                              << (item.second.count - 1 - hz_options_.skip_sample_num) /
                                     std::chrono::duration_cast<std::chrono::duration<double>>((item.second.last_time - item.second.first_time)).count()
                              << std::setw(hz_width) << item.second.max_delta.count() << std::setw(hz_width)
                              << item.second.count - hz_options_.skip_sample_num - 1 << std::setw(hz_width) << "normal" << std::endl;
                    //   std::setw(hz_width) << stdDev
                } else {
                    std::cout << std::left << std::setw(topic_col_width) << item.first << std::fixed << std::setprecision(d_precision) << std::setw(hz_width) << 0 << std::setw(hz_width) << 0 << std::setw(hz_width) << 0
                              << std::setw(hz_width) << "no message" << std::endl;
                }
            }
            std::cout << "\033[" << topic_infos_.size() + 1 << "A"
                      << "\r";
            std::cout.flush();
            //重置windows
            if (hz_options_.window_duration != 0 && std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - _last_print_time).count() >= hz_options_.window_duration) {
                for (auto item : topic_infos_) {
                    topic_infos_[item.first].count = 0;
                    topic_infos_[item.first].min_delta = std::chrono::duration<double>(10.0);
                    topic_infos_[item.first].max_delta = std::chrono::duration<double>(0);
                    item.second.count = 0;
                }
                _last_print_time = currentTime;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\033[" << topic_infos_.size() << "B" << "\033[E";
};

void HzImpl::OnDataAvailable(DataReader* reader) {
    // BAG_LOG_DEBUG << "Hz OnDataAvailable";
    //判断是哪个topic的消息
    std::string topic_name = reader->get_topicdescription()->get_name();
    std::string data_type = reader->get_topicdescription()->get_type_name();

    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lk(topic_infos_mutex_);
        if (hz_options_.skip_sample_num >= topic_infos_[topic_name].count) {
            //跳过帧和第一次帧初始化
            topic_infos_[topic_name].count = topic_infos_[topic_name].count + 1;
            topic_infos_[topic_name].first_time = currentTime;
            topic_infos_[topic_name].last_time = currentTime;
        } else {
            std::chrono::duration<double> timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - topic_infos_[topic_name].last_time);
            if (timeSpan.count() < topic_infos_[topic_name].min_delta.count()) {
                topic_infos_[topic_name].min_delta = timeSpan;  // min
            }
            if (topic_infos_[topic_name].max_delta.count() < timeSpan.count()) {
                topic_infos_[topic_name].max_delta = timeSpan;  // max
            }
            topic_infos_[topic_name].last_time = currentTime;
            topic_infos_[topic_name].count = topic_infos_[topic_name].count + 1;
        }
    }
    //take message
    SampleInfo info;
    if ("ProtoMethodBase" == data_type) {
        ProtoMethodBase proto_methodBase;
        reader->take_next_sample(&proto_methodBase, &info);
    } else if ("CmProtoBuf" == data_type) {
        CmProtoBuf temp_cmProtoBuf;
        ReturnCode_t ret = reader->take_next_sample(&(temp_cmProtoBuf), &info);
        if (hozon::netaos::data_tool_common::IsLiteInfoTopic(topic_name) && ret == ReturnCode_t::RETCODE_OK) {
            //adf-lite info topic, 获取topic里面的hz信息
            if (info.valid_data) {
                hozon::adf::lite::dbg::FreqDebugMessage prototype_msg;
                prototype_msg.ParseFromArray(temp_cmProtoBuf.str().data(), temp_cmProtoBuf.str().size());
                std::lock_guard<std::mutex> lk(topic_infos_mutex_);

                for (int i = 0; i < prototype_msg.mutable_elements()->size(); ++i) {
                    auto ele = prototype_msg.mutable_elements()->at(i);
                    topic_infos_[ele.topic()].last_time = currentTime;
                    topic_infos_[ele.topic()].alit_fre = ele.freq();
                    topic_infos_[ele.topic()].max_delta = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::microseconds(ele.max_delta_us()));
                    topic_infos_[ele.topic()].min_delta = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::microseconds(ele.min_delta_us()));
                    topic_infos_[ele.topic()].count = ele.samples();
                    topic_infos_[ele.topic()].is_matched = true;
                }
            }
        }
    } else if ("CmSomeipBuf" == data_type) {
        CmSomeipBuf temp_cmSomeipBuf;
        ReturnCode_t ret = reader->take_next_sample(&(temp_cmSomeipBuf), &info);
        if (hozon::netaos::data_tool_common::IsLiteInfoTopic(topic_name) && ret == ReturnCode_t::RETCODE_OK) {
            //adf-lite info topic, 获取topic里面的hz信息
            if (info.valid_data) {
                hozon::adf::lite::dbg::FreqDebugMessage prototype_msg;
                prototype_msg.ParseFromArray(temp_cmSomeipBuf.str().data(), temp_cmSomeipBuf.str().size());
                std::lock_guard<std::mutex> lk(topic_infos_mutex_);

                for (int i = 0; i < prototype_msg.mutable_elements()->size(); ++i) {
                    auto ele = prototype_msg.mutable_elements()->at(i);
                    topic_infos_[ele.topic()].last_time = currentTime;
                    topic_infos_[ele.topic()].alit_fre = ele.freq();
                    topic_infos_[ele.topic()].max_delta = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::microseconds(ele.max_delta_us()));
                    topic_infos_[ele.topic()].min_delta = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::microseconds(ele.min_delta_us()));
                    topic_infos_[ele.topic()].count = ele.samples();
                    topic_infos_[ele.topic()].is_matched = true;
                }
            }
        }
    } else if (data_type.find("ZeroCopyImg8M420") != std::string::npos) {
        // ZeroCopyImg8M420 temp_zerocopy;
        // ReturnCode_t ret = reader->take_next_sample(&(temp_zerocopy), &info);
        // if (ret == ReturnCode_t::RETCODE_OK) {
        // }

        FASTDDS_SEQUENCE(ZeroCopyImg8M420Seq, ZeroCopyImg8M420);
        ZeroCopyImg8M420Seq data;
        SampleInfoSeq infos;

        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg8M420& sample = data[i];

                    // TOPIC_LOG_INFO << "---------- Sample received " << data_type
                    //         << " at address " << &sample
                    //         << (reader->is_sample_valid(&sample, &infos[i]) ? " is valid" : " was replaced" )
                    //         << "  seq = " << sample.header().frame_count()
                    //         << "  length = " << static_cast<int32_t>(sample.length());
                }
            }

            reader->return_loan(data, infos);
        }
    } else if (data_type.find("ZeroCopyImg2M422") != std::string::npos) {
        // ZeroCopyImg2M422 temp_zerocopy;
        // ReturnCode_t ret = reader->take_next_sample(&(temp_zerocopy), &info);
        // if (ret == ReturnCode_t::RETCODE_OK) {
        // }

        FASTDDS_SEQUENCE(ZeroCopyImg2M422Seq, ZeroCopyImg2M422);
        ZeroCopyImg2M422Seq data;
        SampleInfoSeq infos;

        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg2M422& sample = data[i];
                    // TOPIC_LOG_INFO << "---------- Sample received " << data_type
                    //         << " at address " << &sample
                    //         << (reader->is_sample_valid(&sample, &infos[i]) ? " is valid" : " was replaced" )
                    //         << "  seq = " << sample.header().frame_count()
                    //         << "  length = " << static_cast<int32_t>(sample.length());
                }
            }

            reader->return_loan(data, infos);
        }
    } else if (data_type.find("ZeroCopyImg3M422") != std::string::npos) {
        // ZeroCopyImg3M422 temp_zerocopy;
        // ReturnCode_t ret = reader->take_next_sample(&(temp_zerocopy), &info);
        // if (ret == ReturnCode_t::RETCODE_OK) {
        // }

        FASTDDS_SEQUENCE(ZeroCopyImg3M422Seq, ZeroCopyImg3M422);
        ZeroCopyImg3M422Seq data;
        SampleInfoSeq infos;

        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg3M422& sample = data[i];
                    // TOPIC_LOG_INFO << "---------- Sample received " << data_type
                    //         << " at address " << &sample
                    //         << (reader->is_sample_valid(&sample, &infos[i]) ? " is valion_subscription_matchedd" : " was replaced" )
                    //         << "  seq = " << sample.header().frame_count()
                    //         << "  length = " << static_cast<int32_t>(sample.length());
                }
            }

            reader->return_loan(data, infos);
        }
    } else {
    }
}

void HzImpl::OnSubscribed(TopicInfo topic_info) {
    std::lock_guard<std::mutex> lk(topic_infos_mutex_);
    topic_infos_[topic_info.topicName];
    topic_infos_[topic_info.topicName].is_matched = true;
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon
