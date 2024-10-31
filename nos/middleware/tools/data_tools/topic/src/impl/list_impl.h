#pragma once
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/types/DynamicDataFactory.h>
#include "list.h"
#include "sub_base.h"

namespace hozon {
namespace netaos {
namespace topic {

class ListImpl : public SubBase {
   public:
    ListImpl();
    virtual ~ListImpl();
    void Start(ListOptions list_options);
    void Stop();
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon