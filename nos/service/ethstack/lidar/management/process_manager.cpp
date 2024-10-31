#include "management/process_manager.h"
#include <string>
#include "process_manager.h"
#include "publish/point_cloud_pub.h"
#include "third_party/orin/jsoncpp/include/json/value.h"




namespace hozon {
namespace ethstack {
namespace lidar {

    static std::mutex  s_mutex;
    ProcessManager* ProcessManager::s_instance = nullptr;

    ProcessManager& ProcessManager::Instance()
    {
        if (nullptr == s_instance) {
            std::lock_guard<std::mutex> lock(s_mutex);
            if (nullptr == s_instance) {
                s_instance = new ProcessManager();
            }
        }
        return *s_instance;
    }

    ProcessManager::ProcessManager()
        : udp_pointcloud(nullptr)
        , m_ifName("")
        , m_lidarFrame("")
        , m_localAddr("")
        , stop_(false)
    {
    }

    ProcessManager::~ProcessManager()
    {
    }

    void ProcessManager::Init()
    {
        {
            LIDAR_LOG_INFO << "Hesai Lidar UDP connect info matrix request socket... ";
            LIDAR_LOG_INFO << "Lidar point cloud udp multicast data socket... ";
            // Lidar point cloud udp multicast data.
            EthernetSocketInfo pointCloudSocket;
            pointCloudSocket.frame_id =  m_lidarFrame;
            pointCloudSocket.if_name = m_ifName;
            pointCloudSocket.local_ip = LOCAL_DEV_IP;
            pointCloudSocket.local_port = LOCAL_POINT_CLOUD_PORT;
            pointCloudSocket.remote_ip = LIDAR_DEV_ADDRESS;
            pointCloudSocket.remote_port = REMOTE_POINT_CLOUD_PORT;
            pointCloudSocket.multicast = POINT_CLOUD_MULTICAST_ADDRESS;
            udp_pointcloud = std::make_shared<UdpBase>(pointCloudSocket);
        }

        {
            LIDAR_LOG_INFO << "Lidar fault message udp multicast data socket... ";
            // Lidar fault message udp multicast data.
            EthernetSocketInfo faultMessageSocket;
            faultMessageSocket.frame_id =  m_lidarFrame;
            faultMessageSocket.if_name = m_ifName;
            faultMessageSocket.local_ip = LOCAL_DEV_IP;
            faultMessageSocket.local_port = LOCAL_FAULT_MESSAGE_PORT;
            faultMessageSocket.remote_ip = LIDAR_DEV_ADDRESS;
            faultMessageSocket.remote_port = REMOTE_FAULT_MESSAGE_PORT;
            faultMessageSocket.multicast = POINT_CLOUD_MULTICAST_ADDRESS;
            udp_fault_message = std::make_shared<UdpBase>(faultMessageSocket); 
        }
        
        PointCloudPub::Instance().Init("/soc/pointcloud");
        PointCloudPub::Instance().lidarStatusReportFunction();
        PointCloudPub::Instance().Pub();                                
        LIDAR_LOG_INFO << "Lidar ProcessManager() Init Complete.";
    }

    void ProcessManager::Start()
    {
        LIDAR_LOG_INFO << "Lidar ProcessManager Start()";
        stop_ = false;
        udp_pointcloud->Start();
        udp_pointcloud->StartProcess();

        udp_fault_message->Start();
        udp_fault_message->StartProcess();
        LIDAR_LOG_INFO << "Lidar ProcessManager Thread Create .";

    }


    void ProcessManager::Stop()
    {
        LIDAR_LOG_INFO << "ProcessManager Stop()";
        stop_ = true;
        udp_pointcloud->Stop();
        udp_fault_message->Stop();
        PointCloudPub::Instance().Stop();

    }

    void ProcessManager::SetIfName(const std::string& ifName)
    {
        m_ifName = ifName;
    }

    void ProcessManager::SetLidarFrame(const std::string& lidarFrame)
    {
        m_lidarFrame = lidarFrame;
    }

}
}
}
