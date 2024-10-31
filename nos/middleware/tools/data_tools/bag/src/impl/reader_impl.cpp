#include "impl/reader_impl.h"
#include <sys/stat.h>
#include <fstream>
#include <json/json.h>
#include "ament_index_cpp/get_search_paths.hpp"
#include "data_tools_logger.hpp"
#include "impl/complete_pcd.h"
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include "rcpputils/filesystem_helper.hpp"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace bag {

ReaderImpl::ReaderImpl() {
    reader_ = std::make_unique<rosbag2_cpp::Reader>();
    GetTopicListFromFileJson();
};

ReaderImpl::~ReaderImpl(){};

ReaderErrorCode ReaderImpl::Open(const std::string uri, const std::string storage_id) {
    ReaderErrorCode ret = ReaderErrorCode::READ_FILE_FAILED;
    if (rcpputils::fs::is_directory(rcpputils::fs::path(uri))) {
        BAG_LOG_ERROR << uri << " should be a xxx.cmap file , but it is a folder.";
        return ret;
    }
    rosbag2_storage::StorageOptions storage_option;
    storage_option.uri = uri;
    storage_option.storage_id = storage_id;
    rosbag2_cpp::ConverterOptions converter_options{};
    if (!reader_->open(storage_option, converter_options)) {
        CONCLE_BAG_LOG_ERROR << "open " << storage_option.uri << " failed.";
        return ReaderErrorCode::READ_FILE_FAILED;
    }

    std::vector<rosbag2_storage::TopicMetadata> topic_meta_data = reader_->get_all_topics_and_types();
    for (auto temp : topic_meta_data) {
        topic_type_map_[temp.name] = temp.type;
    }
    return ReaderErrorCode::SUCCESS;
}

void ReaderImpl::Close() {
    reader_->close();
}

bool ReaderImpl::HasNext() {
    return reader_->has_next();
}

std::shared_ptr<rosbag2_storage::SerializedBagMessage> ReaderImpl::ReadNext() {
    return reader_->read_next();
};

std::map<std::string, std::string> ReaderImpl::GetAllTopicsAndTypes() const {
    return topic_type_map_;
};

void ReaderImpl::SetFilter(const std::vector<std::string>& topics) {
    rosbag2_storage::StorageFilter storage_filter;
    storage_filter.topics = topics;
    reader_->set_filter(storage_filter);
};

void ReaderImpl::ResetFilter() {
    reader_->reset_filter();
};

void ReaderImpl::Seek(const int64_t& timestamp) {
    reader_->seek(timestamp);
};

// reader_interfaces::BaseReaderInterface& get_implementation_handle() const { return *reader_impl_; }

void ReaderImpl::ToPcd(hozon::soc::PointCloud& msg, std::vector<uint8_t>& pcd_message) {
    PPointCloud::Ptr right_cloud_pcl(new PPointCloud);
    //Hesai lidar
    for (uint32_t i = 0; i < msg.points_size(); i++) {
        const auto& pt = msg.points(i);
        LPoint tmp_pt;
        tmp_pt.time = pt.time();
        tmp_pt.x = pt.x();
        tmp_pt.y = pt.y();
        tmp_pt.z = pt.z();
        tmp_pt.intensity = pt.intensity();
        tmp_pt.ring = pt.ring();
        tmp_pt.block = pt.block();
        tmp_pt.distance = pt.distance();
        tmp_pt.pitch = pt.pitch();
        tmp_pt.yaw = pt.yaw();
        tmp_pt.label = pt.label();
        // lidarCloudIn->points[i] = point;
        right_cloud_pcl->points.push_back(tmp_pt);
    }
    right_cloud_pcl->width = right_cloud_pcl->points.size();
    right_cloud_pcl->height = 1;
    // save image
    std::string result_image_path_ = "/tmp/temp_lidar.active";
    // 检查目录是否存在
    struct stat info;
    std::string tmp_path = "tmp";
    if (stat(tmp_path.c_str(), &info) != 0) {
        result_image_path_ = "./temp_lidar.active";
    }

    pcl::PCDWriter writer;
    writer.write(result_image_path_, *right_cloud_pcl, true);
    // 打开文件并以二进制形式读取内容
    std::ifstream infile(result_image_path_, std::ios::in | std::ios::binary);
    if (infile.is_open()) {
        // 确定文件大小
        infile.seekg(0, std::ios::end);
        std::streampos fileSize = infile.tellg();
        infile.seekg(0, std::ios::beg);

        // 读取文件内容到std::vector<uint8_t>
        pcd_message.resize(fileSize);
        infile.read(reinterpret_cast<char*>(pcd_message.data()), fileSize);
        // 关闭文件
        infile.close();
    } else {
        BAG_LOG_ERROR << "Error: Failed to open file for reading.";
        return;
    }
}

void ReaderImpl::ToJpg(const std::string topic_name, hozon::soc::CompressedImage& msg, std::vector<uint8_t>& jpg_message) {
    if (TARGET_PLATFORM == "x86_2004") {
        std::vector<uint8_t> temp_msg(msg.data().begin(), msg.data().end());
        if (camera_decoder_map_.end() == camera_decoder_map_.find(topic_name)) {
            std::unique_ptr<Decoder> decoder_uptr_ = DecoderFactory::Create(hozon::netaos::codec::kDeviceType_Cpu);
            hozon::netaos::codec::DecodeInitParam param;
            param.yuv_type = hozon::netaos::codec::YuvType::kYuvType_YUVJ420P;
            param.codec_type = hozon::netaos::codec::CodecType::kCodecType_H265;
            decoder_uptr_->Init(param);
            camera_decoder_map_[topic_name] = std::move(decoder_uptr_);
        }
        if (nullptr != camera_decoder_map_[topic_name]) {
            hozon::netaos::codec::CodecErrc res = camera_decoder_map_[topic_name]->Process(temp_msg, jpg_message);
            if (res != hozon::netaos::codec::kDecodeSuccess) {
                if (res == hozon::netaos::codec::kDecodeInvalidFrame) {
                    std::cout << "\033[1;33m"
                              << "invalid frame type."
                              << "\033[0m" << std::endl;
                } else {
                    BAG_LOG_ERROR << "failed to encode the frame.";
                }
            }
        } else {
            BAG_LOG_ERROR << "creat decoder failed";
        }
    } else {
        BAG_LOG_ERROR << "h265 to jpg only support in x86 platform.";
    }
}

bool ReaderImpl::GetTopicListFromFileJson() {
    bool ret = true;
    std::string mapFilePath = "";
    auto paths = ament_index_cpp::get_search_paths();
    for (auto path : paths) {
        auto temp_path = path + "/topic_classification_config.json";
        struct stat s;
        if (stat(temp_path.c_str(), &s) == 0) {
            if (s.st_mode & S_IFREG) {
                // Regular file
                mapFilePath = temp_path;
                break;
            }
        }
    }
    if ("" == mapFilePath) {
        BAG_LOG_ERROR << "can't find topic_classification_config.json in conf/bag. Please ensure that AMENT_PREFIX_PATH are set correctly.";
        return false;
    }
    std::ifstream in(mapFilePath, std::ios::binary);
    if (!in.is_open()) {
        BAG_LOG_ERROR << "Error opening " + mapFilePath;
        return false;
    }
    std::string str;
    copy(std::istream_iterator<unsigned char>(in), std::istream_iterator<unsigned char>(), back_inserter(str));
    Json::CharReaderBuilder builder;
    Json::CharReader* reader(builder.newCharReader());
    Json::Value root;
    JSONCPP_STRING errs;
    if (reader->parse(str.c_str(), str.c_str() + str.length(), &root, &errs)) {
        if (!root.isNull()) {
            Json::Value list1 = root["pcd_topic_list"];
            Json::Value list2 = root["jpg_topic_list"];
            for (const auto& element : list1) {
                pcd_topic_list_.push_back(element.asString());
            }
            for (const auto& element : list2) {
                jpg_topic_list_.push_back(element.asString());
            }
        } else {
            BAG_LOG_ERROR << "topic_classification_config.json is empty";
            return false;
        }
    } else {
        BAG_LOG_ERROR << "read topic_classification_config.json failed";
        ret = false;
    }
    in.close();
    return ret;
}

rosbag2_storage::BagMetadata ReaderImpl::GetMetadata() {
    rosbag2_storage::BagMetadata data = reader_->get_metadata();
    data.app_version = paese_app_version(data.app_version);
    return data;
}

std::string ReaderImpl::paese_app_version(const std::string& app_version_str) {
    if ("" != app_version_str) {
        Json::CharReaderBuilder builder;
        Json::CharReader* reader = builder.newCharReader();

        Json::Value root;
        std::string errors;

        // 解析 JSON 字符串
        bool parsingSuccessful = reader->parse(app_version_str.c_str(), app_version_str.c_str() + app_version_str.size(), &root, &errors);

        delete reader;
        if (!parsingSuccessful) {
            COMMON_LOG_ERROR << "/app/version.json parse failed!" << errors;
            return "";
        }

        // 判断是哪种类型的 JSON 字符串
        if (root.isObject()) {
            if (root.isMember("ORIN") && root["ORIN"].isObject()) {
                return "HZ: , NOS: " + root["ORIN"]["EP41"]["middleWare"].asString();
            } else if (root.isMember("NOS")) {
                return "HZ: " + root["HZ"].asString() + ", NOS: " + root["NOS"].asString();
            }
        } else {
            COMMON_LOG_ERROR << "/app/version.json form unknow.";
        }
    }
    return "";
}

void ReaderImpl::get_all_attachments_filepath(std::vector<std::string>& attach_list) {
    reader_->get_all_attachments_filepath(attach_list);
}

std::shared_ptr<rosbag2_storage::Attachment> ReaderImpl::read_attachment(const std::string name) {
    return reader_->read_Attachment(name);
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
