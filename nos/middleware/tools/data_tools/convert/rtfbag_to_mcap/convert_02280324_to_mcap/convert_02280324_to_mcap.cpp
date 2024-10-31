#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <json/json.h>
#include "ament_index_cpp/get_search_paths.hpp"
#include "convert_base.h"
#include "data_tools_logger.hpp"

// #include "mdc_struct_2_proto_message.hpp"
#include "rtf/rtfbag/RtfBagReader.h"
#include "vrtf/vcc/serialize/dds_serialize.h"

#include "convert_code/CameraEncoded_to_ImageOut.hpp"
#include "convert_code/ChassisInfoFrame_to_ChassisOut.hpp"
// #include "convert_code/ChassisInfoFrame_to_FreeSpaceOut.hpp"
#include "convert_code/FreeSpaceFrame_to_FreeSpaceOut.hpp"
#include "convert_code/ImuInsFrame_to_ImuInsOut.hpp"
#include "convert_code/LaneLineFrame_to_TransportElement.hpp"
#include "convert_code/LocationFrame_to_LocalizationOut.hpp"
#include "convert_code/LocationNodeInfo_to_HafNodeInfo.hpp"
#include "convert_code/ObjectCameraFrame_to_ObstacleOut.hpp"
#include "convert_code/ObjectFusionFrame_to_ObstacleOut.hpp"
#include "convert_code/ObjectLidarFrame_to_ObstacleOut.hpp"
#include "convert_code/ParkingLotFrame_to_ParkingLotOut.hpp"
#include "convert_code/PointCloudFrame_to_PointCloudOut.hpp"
#include "convert_code/RadarTrackArrayFrame_to_RadarOut.hpp"
#include "convert_code/StateMachineFrame_to_StateMachineOut.hpp"
#include "hozon/chassis/impl_type_chassisinfoframe.h"  //mdc 数据变量
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "impl/convert_impl.h"
#include "proto/perception/perception_freespace.pb.h"  //proto 数据变量

using namespace rtf::rtfbag;
using namespace hozon::netaos::data_tool_common;

template <typename MdcStructType, typename ProtoMessageType>
using ConvertFunc = std::function<ProtoMessageType(MdcStructType)>;

template <typename MdcStructType, typename ProtoMessageType>
void GetProtoSerializedData(const std::vector<std::uint8_t>& mdc_buff, std::vector<std::uint8_t>& proto_buff, ConvertFunc<MdcStructType, ProtoMessageType> func) {
    //反序列化，获取mdc对象
    vrtf::serialize::dds::Deserializer<MdcStructType> deserializer(mdc_buff.data(), mdc_buff.size());
    MdcStructType mdc_struct_data = deserializer.GetValue();
    //mdc对象转换成proto对象
    ProtoMessageType message = func(mdc_struct_data);
    //获取proto对象序列化数据
    std::string serialized_data;
    if (!message.SerializeToString(&serialized_data)) {
        BAG_LOG_ERROR << message.GetTypeName() << " serialize to string failed!";
    }

    CmProtoBuf proto_idl_data;
    proto_idl_data.name() = message.GetTypeName();
    proto_idl_data.str().assign(serialized_data.begin(), serialized_data.end());

    eprosima::fastrtps::rtps::SerializedPayload_t payload;
    CmProtoBufPubSubType sub_type;
    payload.reserve(sub_type.getSerializedSizeProvider(&proto_idl_data)());
    sub_type.serialize(&proto_idl_data, &payload);
    proto_buff.resize(payload.length);
    std::memcpy(proto_buff.data(), payload.data, payload.length);
}

class Convert_02280324_ToMcap : public ConvertBase {
   private:
    std::function<void(std::string, std::string, int64_t time, std::vector<std::uint8_t>)> _callbackFunction;
    std::pair<std::string, std::string> ReadFileJson(std::string eventName);
    std::vector<std::uint8_t> ConvertMessage(RtfBagReader::EventMsg event);

   public:
    void RegistMessageCallback(std::function<void(std::string, std::string, int64_t time, std::vector<std::uint8_t>)> callbackFunction);
    void Convert(std::string input_file, const std::vector<std::string>& exclude_topics, const std::vector<std::string>& topics);
    Convert_02280324_ToMcap() : _callbackFunction(nullptr){};
    ~Convert_02280324_ToMcap(){};
};

void Convert_02280324_ToMcap::Convert(std::string input_file, const std::vector<std::string>& exclude_topics, const std::vector<std::string>& topics) {
    // 初始化rtfReader
    RtfBagReader reader(input_file);
    reader.Init();
    std::vector<std::string> channel_name_list;
    auto bag_info = reader.GetBagInfo();
    std::string bag_version = bag_info.GetBagVersion();

    for (auto info : bag_info.GetEventInfoList()) {
        if ("" == ReadFileJson(info.first).first) {
            BAG_LOG_WARN << "not configured for " << info.first << " in topic_info_mapping.json. will be pass";
        };
    }

    //循环读取rtfbag中的序列化数据
    RtfBagReader::EventMsg event;
    while (reader.ReadEventMsg(event) != RtfBagReader::ErrorCode::WRONG_READ) {
        if (hozon::netaos::bag::ConvertImpl::is_stop) {
            break;
        }

        //过滤不需转换的topic
        std::string eventName = event.GetEventName();
        if (topics.size() > 0 && (topics.end() == std::find(topics.begin(), topics.end(), eventName))) {
            continue;
        } else if (exclude_topics.size() > 0 && (exclude_topics.end() != std::find(exclude_topics.begin(), exclude_topics.end(), eventName))) {
            continue;
        }

        //转换数据
        std::vector<uint8_t> item = ConvertMessage(event);
        std::pair<std::string, std::string> topic_info = ReadFileJson(event.GetEventName());
        if ("" != topic_info.first) {
            if (item.size() > 0) {
                if (nullptr != _callbackFunction) {
                    //返回转换后的数据
                    _callbackFunction(topic_info.first, topic_info.second, event.GetTimeStamp() * 1000000, item);
                }
            } else {
                BAG_LOG_WARN << "message serialized data is empty, event name = " << event.GetEventName();
            }

        } else {
            // BAG_LOG_WARN << "not configured for " << event.GetEventName() << " in topic_info_mapping.json. pass";
        }
    }
}

std::vector<std::uint8_t> Convert_02280324_ToMcap::ConvertMessage(RtfBagReader::EventMsg event) {
    std::string eventName = event.GetEventName();
    std::vector<std::uint8_t> proto_buff;
    if ("/Hozon/ServiceInterface/HozonInterface_Chassis/hozonEvent[/Hozon/ModuleBase/Chassis/Service/DDS/Provider/ChassisProviderInstance_Chassis_DDS_1]" == eventName) {
        GetProtoSerializedData<hozon::chassis::ChassisInfoFrame, hozon::soc::Chassis>(event.GetBuffer(), proto_buff, ChassisInfoFrameToChassisOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_ImuInfo/hozonEvent[/hozon/PlatformApplication/InsPvatbApplication/Service/Provider/ImuToAppProvidedDdsServiceInstance]" == eventName) {
        GetProtoSerializedData<hozon::sensors::ImuInsInfoFrame, hozon::soc::ImuIns>(event.GetBuffer(), proto_buff, ImuInsInfoFrameToImuInsOut);
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_71]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_73]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_75]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_76]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_77]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_78]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_79]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_80]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_81]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_82]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if (
        "/HuaweiMDC/PlatformServiceInterface/CameraServiceInterfacePkg/CameraEncodedMbufServiceInterface/cameraEncodedMbufEvent[/HuaweiMDC/PlatformApplication/CameraVencApplication/"
        "CameraVencService/"
        "CameraServiceProvider/camera_venc_encoded_mbuf_83]" == eventName) {
        GetProtoSerializedData<CameraEncodedMbufStructHelp, hozon::soc::CompressedImage>(event.GetBuffer(), proto_buff, CameraEncodedFrameToImageOut);
        // image
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCamFusion_DDS_1]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/fisheye_perception/Service/DDS/Provider/fisheye_perceptionProviderInstance_objcamera_DDS_104]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCam_DDS_2]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCam_DDS_3]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCam_DDS_4]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCam_DDS_5]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCam_DDS_6]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCam_DDS_7]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Camera/hozonEvent[/Hozon/ModuleBase/PerceptionCameraObject/Service/DDS/Provider/PerCameraObjectProviderInstance_ObjCam_DDS_8]" ==
               eventName) {
        GetProtoSerializedData<hozon::object::ObjectCameraFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjCameraFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Lidar/hozonEvent[/Hozon/ModuleBase/PerceptionLidar/Service/DDS/Provider/PerLidarProviderInstance_ObjLidar_DDS_10]" == eventName) {
        GetProtoSerializedData<hozon::object::ObjectLidarFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjLidarFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Lidar/hozonEvent[/Hozon/ModuleBase/uss_perception/Service/DDS/Provider/uss_perceptionProviderInstance_Lidar_DDS_100]" == eventName) {
        GetProtoSerializedData<hozon::object::ObjectLidarFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjLidarFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Fusion/hozonEvent[/Hozon/ModuleBase/PerceptionFusion/Service/DDS/Provider/PerFusionProviderInstance_ObjFusion_DDS_1]" == eventName) {
        GetProtoSerializedData<hozon::object::ObjectFusionFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjFusionFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Fusion/hozonEvent[/Hozon/ModuleBase/parking_fusion/Service/DDS/Provider/obstacle_fusionProviderInstance_objfusion_DDS_11]" == eventName) {
        GetProtoSerializedData<hozon::object::ObjectFusionFrame, hozon::perception::PerceptionObstacles>(event.GetBuffer(), proto_buff, ObjFusionFrameToObstacleOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_StateMachine/hozonEvent[/Hozon/ModuleBase/Chassis/Service/DDS/Provider/ChassisProviderInstance_StateMachine_DDS_11]" == eventName) {
        GetProtoSerializedData<hozon::statemachine::StateMachineFrame, hozon::state::StateMachine>(event.GetBuffer(), proto_buff, StateMachineFrameToStateMachineOut);
    } else if (
        "/Hozon/ServiceInterface/HozonInterface_StateMachine/hozonEvent[/Hozon/ModuleBase/PerceptionStateMachine/Service/DDS/Provider/PerceptionStateMachineProviderInstance_StateMachine_DDS_12]" ==
        eventName) {
        GetProtoSerializedData<hozon::statemachine::StateMachineFrame, hozon::state::StateMachine>(event.GetBuffer(), proto_buff, StateMachineFrameToStateMachineOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_StateMachine/hozonEvent[/Hozon/ModuleBase/StateMachine/Service/DDS/Provider/StateMachineProviderInstance_StateMachine_DDS_1]" == eventName) {
        GetProtoSerializedData<hozon::statemachine::StateMachineFrame, hozon::state::StateMachine>(event.GetBuffer(), proto_buff, StateMachineFrameToStateMachineOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_StateMachine/hozonEvent[/Hozon/ModuleBase/StateMachine/Service/DDS/Provider/StateMachineProviderInstance_StateMachine_DDS_2]" == eventName) {
        GetProtoSerializedData<hozon::statemachine::StateMachineFrame, hozon::state::StateMachine>(event.GetBuffer(), proto_buff, StateMachineFrameToStateMachineOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_PointCloud/hozonEvent[/Service/DDS/Provider/LidarDdsProvidedServiceInstance_1]" == eventName) {
        GetProtoSerializedData<hozon::sensors::PointCloudFrame, hozon::soc::PointCloud>(event.GetBuffer(), proto_buff, PointCloudFrameToPointCloudOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_PointCloud/hozonEvent[/Service/DDS/Provider/LidarDdsProvidedServiceInstance_2]" == eventName) {
        GetProtoSerializedData<hozon::sensors::PointCloudFrame, hozon::soc::PointCloud>(event.GetBuffer(), proto_buff, PointCloudFrameToPointCloudOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Location/hozonEvent[/Hozon/ModuleBase/hz_location/Service/DDS/Provider/hz_location_ProviderInstance_Location_DDS_1]" == eventName) {
        GetProtoSerializedData<hozon::location::LocationFrame, hozon::localization::Localization>(event.GetBuffer(), proto_buff, LocationFrameToLocalizationOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Location/hozonEvent[/Hozon/ModuleBase/parking_slam/Service/DDS/Provider/SLAMProviderInstance_Location_DDS_2]" == eventName) {
        GetProtoSerializedData<hozon::location::LocationFrame, hozon::localization::Localization>(event.GetBuffer(), proto_buff, LocationFrameToLocalizationOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Radar/hozonEvent[/hozon/AngleCornerDriverApplication/Service/Provider/radarAngleServiceInstance_1]" == eventName) {
        GetProtoSerializedData<hozon::sensors::RadarTrackArrayFrame, hozon::soc::RadarTrackArrayFrame>(event.GetBuffer(), proto_buff, RadarFrameToRadarOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Radar/hozonEvent[/hozon/AngleCornerDriverApplication/Service/Provider/radarAngleServiceInstance_2]" == eventName) {
        GetProtoSerializedData<hozon::sensors::RadarTrackArrayFrame, hozon::soc::RadarTrackArrayFrame>(event.GetBuffer(), proto_buff, RadarFrameToRadarOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Radar/hozonEvent[/hozon/AngleCornerDriverApplication/Service/Provider/radarAngleServiceInstance_3]" == eventName) {
        GetProtoSerializedData<hozon::sensors::RadarTrackArrayFrame, hozon::soc::RadarTrackArrayFrame>(event.GetBuffer(), proto_buff, RadarFrameToRadarOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Radar/hozonEvent[/hozon/AngleCornerDriverApplication/Service/Provider/radarAngleServiceInstance_4]" == eventName) {
        GetProtoSerializedData<hozon::sensors::RadarTrackArrayFrame, hozon::soc::RadarTrackArrayFrame>(event.GetBuffer(), proto_buff, RadarFrameToRadarOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Obj_Radar/hozonEvent[/hozon/Service/instance/RadarFrontTrackDdsProvidedServiceInstance]" == eventName) {
        GetProtoSerializedData<hozon::sensors::RadarTrackArrayFrame, hozon::soc::RadarTrackArrayFrame>(event.GetBuffer(), proto_buff, RadarFrameToRadarOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_ParkingLot/hozonEvent[/Hozon/ModuleBase/parking_fusion/Service/DDS/Provider/obstacle_fusionProviderInstance_ParkingLot_DDS_1]" == eventName) {
        GetProtoSerializedData<hozon::parkinglot::ParkingLotFrame, hozon::perception::ParkingLotOutArray>(event.GetBuffer(), proto_buff, ParkingLotFrameToParkingLotOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_ParkingLot/hozonEvent[/Hozon/ModuleBase/uss_perception/Service/DDS/Provider/uss_perceptionProviderInstance_ParkingLot_DDS_100]" == eventName) {
        GetProtoSerializedData<hozon::parkinglot::ParkingLotFrame, hozon::perception::ParkingLotOutArray>(event.GetBuffer(), proto_buff, ParkingLotFrameToParkingLotOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_ParkingLot/hozonEvent[/Hozon/ModuleBase/fisheye_perception/Service/DDS/Provider/fisheye_perceptionProviderInstance_ParkingLot_DDS_104]" ==
               eventName) {
        GetProtoSerializedData<hozon::parkinglot::ParkingLotFrame, hozon::perception::ParkingLotOutArray>(event.GetBuffer(), proto_buff, ParkingLotFrameToParkingLotOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_ParkingLot/hozonEvent[/Hozon/ModuleBase/parking_slam/Service/DDS/Provider/SLAMProviderInstance_ParkingLot_DDS_106]" == eventName) {
        GetProtoSerializedData<hozon::parkinglot::ParkingLotFrame, hozon::perception::ParkingLotOutArray>(event.GetBuffer(), proto_buff, ParkingLotFrameToParkingLotOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Freespace/hozonEvent[/Hozon/ModuleBase/PerceptionLidar/Service/DDS/Provider/PerLidarProviderInstance_FreeSpace_DDS_10]" == eventName) {
        GetProtoSerializedData<hozon::freespace::FreeSpaceFrame, hozon::perception::FreeSpaceOutArray>(event.GetBuffer(), proto_buff, FreeSpaceFrameToFreeSpaceOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Freespace/hozonEvent[/Hozon/ModuleBase/PerceptionCameraLane/Service/DDS/Provider/PerCameraLaneProviderInstance_FreeSpace_DDS_1]" == eventName) {
        GetProtoSerializedData<hozon::freespace::FreeSpaceFrame, hozon::perception::FreeSpaceOutArray>(event.GetBuffer(), proto_buff, FreeSpaceFrameToFreeSpaceOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Freespace/hozonEvent[/Hozon/ModuleBase/RoadMark/Service/DDS/Provider/RoadMarkProviderInstance_FreespaceFusion_DDS_11]" == eventName) {
        GetProtoSerializedData<hozon::freespace::FreeSpaceFrame, hozon::perception::FreeSpaceOutArray>(event.GetBuffer(), proto_buff, FreeSpaceFrameToFreeSpaceOut);
    } else if ("/Hozon/ServiceInterface/HozonInterface_Freespace/hozonEvent[/Hozon/ModuleBase/parking_fusion/Service/DDS/Provider/obstacle_fusionProviderInstance_FreeSpace_DDS_2]" == eventName) {
        GetProtoSerializedData<hozon::freespace::FreeSpaceFrame, hozon::perception::FreeSpaceOutArray>(event.GetBuffer(), proto_buff, FreeSpaceFrameToFreeSpaceOut);
    } else if (
        "/Hozon/ServiceInterface/HozonInterface_LocationNodeInfo/hozonLocationEvent[/Hozon/ModuleBase/hz_location_plugin/Service/DDS/Provider/"
        "hz_location_plugin_ProviderInstance_LocationNodeInfo_DDS_104]" == eventName) {
        GetProtoSerializedData<hozon::location::LocationNodeInfo, hozon::localization::HafNodeInfo>(event.GetBuffer(), proto_buff, LocationNodeInfoToHafNodeInfo);
    } else if ("/Hozon/ServiceInterface/HozonInterface_LaneLine/hozonEvent[/Hozon/ModuleBase/PerceptionCameraLane/Service/DDS/Provider/PerCameraLaneProviderInstance_LaneLine_DDS_1]" == eventName) {
        GetProtoSerializedData<hozon::laneline::LaneLineFrame, hozon::perception::TransportElement>(event.GetBuffer(), proto_buff, LaneLineFrameToTransportElement);
    } else if ("/Hozon/ServiceInterface/HozonInterface_LaneLine/hozonEvent[/Hozon/ModuleBase/PerceptionVision_pipeline/Service/DDS/Provider/PerCameraObjectProviderInstance_LaneLine_DDS_12]" ==
               eventName) {
        GetProtoSerializedData<hozon::laneline::LaneLineFrame, hozon::perception::TransportElement>(event.GetBuffer(), proto_buff, LaneLineFrameToTransportElement);
    } else {
        // BAG_LOG_WARN << "not found event name=" << eventName;
    }

    return proto_buff;
}

// if ("/Hozon/ServiceInterface/HozonInterface_ImuInfo/hozonEvent[/hozon/PlatformApplication/InsPvatbApplication/Service/Provider/ImuToAppProvidedDdsServiceInstance]" == eventName) {
// vrtf::serialize::dds::Deserializer<s0930_1215::hozon::sensors::ImuInfoFrame> deserializer(event.GetBuffer().data(), event.GetBufferSize());
//     s0930_1215::hozon::sensors::ImuInfoFrame data = deserializer.GetValue();

//     for (std::vector<s0930_1215::hozon::sensors::InsInfoFrame>::iterator it = insInfoVec.begin(); it != insInfoVec.end(); it++) {
//         double instime = (*it).gpsSec;
//         double imuTime = data.gpsSec;

//         double result = imuTime - instime;
//         if ((result <= 0.001) && (result >= -0.001)) {
//             // value same
//             s0228_0324::hozon::sensors::ImuInsInfoFrame newData = combineImuIns(data, (*it));
//             insInfoVec.erase(it);

//             vrtf::serialize::dds::Serializer<s0228_0324::hozon::sensors::ImuInsInfoFrame> serializer(newData);
//             std::uint8_t c[serializer.GetSize()];
//             serializer.Serialize(c);
//             std::vector<std::uint8_t> tempbuff(c, c + serializer.GetSize());
//             buff = tempbuff;
//             return buff;
//         }
//     }

//     // no matched imu info, record insinfo
//     imuInfoVec.push_back(data);
//     return buff;
// }

void Convert_02280324_ToMcap::RegistMessageCallback(std::function<void(std::string, std::string, int64_t time, std::vector<std::uint8_t>)> callbackFunction) {
    _callbackFunction = callbackFunction;
}

std::pair<std::string, std::string> Convert_02280324_ToMcap::ReadFileJson(std::string eventName) {
    std::pair<std::string, std::string> item("", "");
    std::string mapFilePath = "";
    auto paths = ament_index_cpp::get_search_paths();
    for (auto path : paths) {
        auto temp_path = path + "/convert/rtfbag_to_mcap/topic_info_mapping.json";
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
        throw std::runtime_error("can't find topic_info_mapping.json in " + mapFilePath + ". Please ensure that AMENT_PREFIX_PATH are set correctly.");
    }
    std::ifstream in(mapFilePath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Error opening " + mapFilePath);
    }

    std::string str;
    copy(std::istream_iterator<unsigned char>(in), std::istream_iterator<unsigned char>(), back_inserter(str));
    Json::CharReaderBuilder builder;
    Json::CharReader* reader(builder.newCharReader());
    Json::Value root;
    JSONCPP_STRING errs;
    if (reader->parse(str.c_str(), str.c_str() + str.length(), &root, &errs)) {
        const Json::Value info = root[eventName];
        if (!info.isNull()) {
            item.first = root[eventName]["topicName"].asString();
            item.second = root[eventName]["dataType"].asString();
            return item;
        }
    }
    in.close();
    return item;
}

extern "C" ConvertBase* CreatConverter() {
    ConvertBase* temp = new Convert_02280324_ToMcap();
    return temp;
}

extern "C" void DestroyConverter(ConvertBase* ptr) {
    if (nullptr != ptr) {
        Convert_02280324_ToMcap* temp = static_cast<Convert_02280324_ToMcap*>(ptr);
        delete temp;
        ptr = nullptr;
    }
    return;
}