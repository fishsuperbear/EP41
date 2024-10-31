#include "impl/convert_impl.h"
#include <dlfcn.h>
#include <sys/stat.h>
#include "ament_index_cpp/get_search_paths.hpp"
#include "convert_base.h"
#include "data_tools_logger.hpp"
#include "fastdds/rtps/common/SerializedPayload.h"
#include "reader_impl.h"

namespace hozon {
namespace netaos {
namespace bag {

bool ConvertImpl::is_stop = false;

ConvertImpl::ConvertImpl() : _writer(nullptr), _count(0) {}

ConvertImpl::~ConvertImpl() {}

//接收proto序列化数据，存储成.mcap
void ConvertImpl::WriteMessage(std::string topic_name, std::string data_type, int64_t time, std::vector<std::uint8_t> data) {
    eprosima::fastrtps::rtps::SerializedPayload_t payload;
    payload.reserve(data.size());
    memcpy(payload.data, data.data(), data.size());
    payload.length = data.size();
    _writer->write(payload, topic_name, data_type, time);
    _count++;
    std::cout << "Converting " << _count << " messages...\r";
    std::cout.flush();
    return;
}

void ConvertImpl::WriteBagMessage(std::shared_ptr<rosbag2_storage::SerializedBagMessage> data) {
    std::string topic_name = data->topic_name;
    if ("" == topic_type_info_[topic_name]) {
        BAG_LOG_ERROR << "can't find data type: " << topic_type_info_[topic_name];
        return;
    }

    _writer->write(data, topic_name, topic_type_info_[topic_name]);
    _count++;
    std::cout << "Converting " << _count << " messages...\r";
    std::cout.flush();
    return;
}

void ConvertImpl::Stop() {
    ConvertImpl::is_stop = true;
}

void ConvertImpl::Start(ConvertOptions convert_option) {
    //初始化 writer
    _writer = std::make_unique<rosbag2_cpp::Writer>();
    rosbag2_storage::StorageOptions storageOptions;
    storageOptions.use_time_suffix = convert_option.use_time_suffix;
    storageOptions.uri = convert_option.output_file;
    if ("cyber" == convert_option.output_file_type) {
        storageOptions.storage_id = "record";
    } else {
        storageOptions.storage_id = convert_option.output_file_type;
    }

    rosbag2_cpp::ConverterOptions converter_options{};
    _writer->open(storageOptions, converter_options);

    if ("rtfbag" == convert_option.intput_file_type && "mcap" == convert_option.output_file_type) {
        //加载转化的so
        if (convert_option.output_file_type != "mcap") {
            BAG_LOG_ERROR << "output file type must be mcap";
            return;
        }

        //0430-0613使用与0228相同的头文件
        if ("0430-0613" == convert_option.intput_data_version) {
            convert_option.intput_data_version = "0228-0324";
        }

        std::string plugin_name = "lib" + convert_option.intput_file_type + "-" + convert_option.intput_data_version + "_" + convert_option.output_file_type + ".so";
        void* handle;
        std::string package_path = "";
        auto paths = ament_index_cpp::get_search_paths();
        for (auto path : paths) {
            auto temp_path = path + "/convert/rtfbag_to_mcap/" + plugin_name;
            std::cout << temp_path << std::endl;
            struct stat s;
            if (stat(temp_path.c_str(), &s) == 0) {
                if (s.st_mode & S_IFREG) {
                    // Regular file
                    package_path = temp_path;
                    break;
                }
            }
        }
        if ("" == package_path) {
            BAG_LOG_ERROR << "can't find plugin_name:  " << plugin_name << ". Please ensure that AMENT_PREFIX_PATH are set correctly.";
            return;
        }
        handle = dlopen(package_path.c_str(), RTLD_LAZY);
        if (!handle) {
            BAG_LOG_ERROR << "Failed to load library: " << dlerror();
            return;
        }
        typedef ConvertBase* (*CreatConverterFunc)();
        CreatConverterFunc creatConverter = (CreatConverterFunc)dlsym(handle, "CreatConverter");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            BAG_LOG_ERROR << "Failed to find symbol: " << dlsym_error;
            dlclose(handle);
            return;
        }
        dlsym_error = dlerror();
        typedef void (*DestroyConverterFunc)(ConvertBase*);
        DestroyConverterFunc destroyConverter = (DestroyConverterFunc)dlsym(handle, "DestroyConverter");
        if (dlsym_error) {
            BAG_LOG_ERROR << "Failed to find symbol: " << dlsym_error;
            dlclose(handle);
            return;
        }

        ConvertBase* converter = creatConverter();
        //注册回调,接收转换后的数据
        converter->RegistMessageCallback(std::bind(&ConvertImpl::WriteMessage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
        //开始转换，传入rtfbag的路径
        converter->Convert(convert_option.input_file, convert_option.exclude_topics, convert_option.topics);
        destroyConverter(converter);

        // 卸载 .so 文件
        dlclose(handle);
        BAG_LOG_INFO << "convert successfully!";
    } else if ("mcap" == convert_option.intput_file_type && "cyber" == convert_option.output_file_type) {
        //打开mcap文件
        ReaderImpl reader;
        reader.Open(convert_option.input_file, convert_option.intput_file_type);
        //获取所有的topic type信息
        topic_type_info_ = reader.GetAllTopicsAndTypes();

        //过滤需要转换的topic
        if (convert_option.topics.size() > 0) {
            reader.SetFilter(convert_option.topics);
        } else if (convert_option.exclude_topics.size() > 0) {
            std::vector<std::string> filter_topic;
            for (auto temp : topic_type_info_) {
                if (convert_option.exclude_topics.end() == std::find(convert_option.exclude_topics.begin(), convert_option.exclude_topics.end(), temp.first)) {
                    filter_topic.push_back(temp.first);
                }
            }
            reader.SetFilter(filter_topic);
        }
        //读取、写入message
        while (reader.HasNext() && !is_stop) {
            auto message = reader.ReadNext();
            WriteBagMessage(message);
        }

    } else {
        BAG_LOG_ERROR << "can not convert " << convert_option.intput_file_type << " to " << convert_option.output_file_type << ".";
    }
    return;
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon