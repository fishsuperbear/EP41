#pragma once
#include "save_impl.h"
#include "data_tools_logger.hpp"
#include "reader_impl.h"
#include "proto/soc/sensor_image.pb.h"

namespace hozon {
namespace netaos {
namespace bag {

SaveImpl::SaveImpl() {}

SaveImpl::~SaveImpl() {}

SaveErrorCode SaveImpl::Start(SaveOptions save_option) {
    if (0 == save_option.topics.size()) {
        BAG_LOG_ERROR << "Please select at least one topic";
        return SaveErrorCode::FAILED_NO_TOPIC_SPECIFIED;
    }

    Reader reader;
    if (hozon::netaos::bag::ReaderErrorCode::SUCCESS != reader.Open(save_option.url, "mcap")) {
        return SaveErrorCode::FAILED;
    }
    std::cout << "open " << save_option.url << std::endl;

    // auto topics_info = reader.GetAllTopicsAndTypes();  //获取包中所有的topic和type
    reader.SetFilter(save_option.topics);  //设在过滤器，只读取指定的topic的message
    std::map<std::string, FILE*> fs_map;
    for (auto item : save_option.topics) {
        // 找到最后一个斜杠的位置
        std::string subString = item;
        size_t lastSlashPos = item.find_last_of('/');
        // 如果找到了斜杠
        if (lastSlashPos != std::string::npos) {
            // 使用substr获取从最后一个斜杠位置到字符串结尾的子字符串
            subString = item.substr(lastSlashPos + 1);
        }

        FILE* fpSave;
        fpSave = fopen((subString + ".h265").c_str(), "wb");
        fs_map[item] = fpSave;
    }

    //取出一个message的序列化数据后，反序列化为指定的proto格式
    while (reader.HasNext()) {
        TopicMessage message_vec = reader.ReadNext();
        hozon::soc::CompressedImage proto_data = reader.DeserializeToProto<hozon::soc::CompressedImage>(message_vec);
        fwrite(proto_data.data().c_str(), 1, proto_data.data().length(), fs_map[message_vec.topic]);
        _count++;
        std::cout << "save " << _count << " message.....\r";
        std::cout.flush();
    }

    for (auto item : fs_map) {
        fclose(item.second);
    }

    return SaveErrorCode::SAVE_SUCCESS;
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon