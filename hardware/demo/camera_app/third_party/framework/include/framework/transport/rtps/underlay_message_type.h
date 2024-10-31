#ifndef CYBER_TRANSPORT_RTPS_UNDERLAY_MESSAGE_TYPE_H_
#define CYBER_TRANSPORT_RTPS_UNDERLAY_MESSAGE_TYPE_H_

#include "framework/transport/rtps/underlay_message.h"
#include "fastrtps/TopicDataType.h"

namespace netaos {
namespace framework {
namespace transport {

/*!
 * @brief This class represents the TopicDataType of the type UnderlayMessage
 * defined by the user in the IDL file.
 * @ingroup UNDERLAYMESSAGE
 */
class UnderlayMessageType : public eprosima::fastrtps::TopicDataType {
 public:
  using type = UnderlayMessage;

  UnderlayMessageType();
  virtual ~UnderlayMessageType();
  bool serialize(void* data, SerializedPayload_t* payload);
  bool deserialize(SerializedPayload_t* payload, void* data);
  std::function<uint32_t()> getSerializedSizeProvider(void* data);
  bool getKey(void* data, InstanceHandle_t* ihandle);
  void* createData();
  void deleteData(void* data);
  MD5 m_md5;
  unsigned char* m_keyBuffer;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_RTPS_UNDERLAY_MESSAGE_TYPE_H_
