#include "udp_dispatcher.h"
#include <arpa/inet.h>


// static int udp_pack_num = 0;

namespace hozon {
namespace ethstack {
namespace lidar {



UdpDispatcher::UdpDispatcher() {
}

UdpDispatcher::~UdpDispatcher() {
    m_parserList.clear();
}

int32_t UdpDispatcher::Parse(uint16_t recvPort,  std::shared_ptr<EthernetPacket> packet)
{
    switch (recvPort)
    {
    case LOCAL_POINT_CLOUD_PORT:    
        //接收组播消息的UDP包，判断包的大小
        if(packet->len == 1118){
            PointCloudParser::Instance().Parse(packet->data, packet->len);
        }
        break;
    case LOCAL_FAULT_MESSAGE_PORT:    
        //接收点云错误消息的UDP包，判断包的大小
        if(packet->len == 99){
            FaultMessageParser::Instance().Parse(packet->data, packet->len);
        }
        break;
    default:
        break;
    }
    return 0;
}



}
}
}