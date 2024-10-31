#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <rosbag2_cpp/reader.hpp>
#include "google/protobuf/message.h"
#include "reader.h"

#include "codec/include/codec_def.h"
#include "codec/include/decoder.h"
#include "codec/include/decoder_factory.h"
#include "proto/soc/point_cloud.pb.h"
#include "proto/soc/sensor_image.pb.h"

// #include "codec/src/cpu/decoder_cpu.h"
using hozon::netaos::codec::Decoder;
using hozon::netaos::codec::DecoderFactory;

namespace hozon {
namespace netaos {
namespace bag {

class ReaderImpl final {
   public:
    explicit ReaderImpl();

    ~ReaderImpl();

    /**
   * Throws if file could not be opened.
   * This must be called before any other function is used.
   * The rosbag is automatically closed on destruction.
   *
   * \param uri bag path
   * \param storage_id bag format
   */
    ReaderErrorCode Open(const std::string uri, const std::string storage_id = "mcap");

    /**
   * Closing the reader instance.
   */
    void Close();

    /**
   * Ask whether the underlying bagfile contains at least one more message.
   *
   * \return true if storage contains at least one more message
   * \throws runtime_error if the Reader is not open.
   */
    bool HasNext();

    /**
   * Read next message from storage. Will throw if no more messages are available.
   * The message will be serialized in the format given to `open`.
   *
   * Expected usage:
   * if (writer.has_next()) message = writer.read_next();
   *
   * \return next message in serialized form
   * \throws runtime_error if the Reader is not open.
   */
    std::shared_ptr<rosbag2_storage::SerializedBagMessage> ReadNext();

    /**
   * Ask bagfile for all topics (including their type identifier) that were recorded.
   *
   * \return vector of topics with topic name and type as std::string
   * \throws runtime_error if the Reader is not open.
   */
    std::map<std::string, std::string> GetAllTopicsAndTypes() const;

    /**
   * Set filters to adhere to during reading.
   *
   * \param storage_filter Filter to apply to reading
   * \throws runtime_error if the Reader is not open.
   */
    void SetFilter(const std::vector<std::string>& topics);

    /**
   * Reset all filters for reading.
   */
    void ResetFilter();

    /**
   * Skip to a specific timestamp for reading.
   */
    void Seek(const int64_t& timestamp);

    // reader_interfaces::BaseReaderInterface& get_implementation_handle() const { return *reader_impl_; }

    void ToPcd(hozon::soc::PointCloud& msg, std::vector<uint8_t>& pcd_message);

    void ToJpg(const std::string topic_name, hozon::soc::CompressedImage& msg, std::vector<uint8_t>& pcd_message);

    rosbag2_storage::BagMetadata GetMetadata();

    void get_all_attachments_filepath(std::vector<std::string> &attach_list);

    std::shared_ptr<rosbag2_storage::Attachment> read_attachment(const std::string name);

    std::vector<std::string> pcd_topic_list_;
    std::vector<std::string> jpg_topic_list_;
    bool is_first_handle_raw_point = true;

   private:
    bool GetTopicListFromFileJson();
    std::string paese_app_version(const std::string& app_version_str);
    std::unique_ptr<rosbag2_cpp::Reader> reader_;
    std::map<std::string, std::string> topic_type_map_;
    std::map<std::string, std::unique_ptr<Decoder>> camera_decoder_map_;
    bool jpgInit_ = false;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
