#include "message_process.h"
#include "data_tools_logger.hpp"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include <json/json.h>

namespace hozon {
namespace netaos {
namespace bag {

static std::mutex inst_mutex_;
MessageProcess* MessageProcess::instance_ = nullptr;

MessageProcess& MessageProcess::Instance(const std::string& config_path) {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(inst_mutex_);
        if (nullptr == instance_) {
            instance_ = new MessageProcess(config_path);
        }
    }
    return *instance_;
}

MessageProcess& MessageProcess::Instance(const std::string& uri, const std::string& storage_id) {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(inst_mutex_);
        if (nullptr == instance_) {
            instance_ = new MessageProcess(uri, storage_id);
        }
    }
    return *instance_;
}

MessageProcess::MessageProcess(const std::string& config_path) {
    lidar_correction_path = config_path + "ATD128P.dat";
    lidar_extrinsics_path = config_path + "lidar_extrinsics.yaml";
    init();
}

MessageProcess::MessageProcess(const std::string& uri, const std::string& storage_id) {
    rosbag2_storage::StorageOptions storage_option;
    storage_option.uri = uri;
    storage_option.storage_id = storage_id;
    rosbag2_cpp::ConverterOptions converter_options{};
    _reader = std::make_unique<rosbag2_cpp::Reader>();
    _reader->open(storage_option, converter_options);
    init();
}

void MessageProcess::init() {
    // _reader = std::make_unique<rosbag2_cpp::Reader>();
    //加载校正文件
    if (loadCorrectionFile(lidar_correction_path) == true) {
        BAG_LOG_INFO << "success load correction file!";
        if_load_correction_ = true;
    } else {
        BAG_LOG_WARN << "fail to load correction file!";
    }

    // 加载外参
    if (LoadExtrinsics(lidar_extrinsics_path) == true) {
        BAG_LOG_INFO << "success load extrinsics file!";
        if_load_extrinsics_ = true;
    } else {
        BAG_LOG_WARN << "fail to load extrinsics file!";
    }
}

MessageProcess::~MessageProcess() {}

bool MessageProcess::LoadExtrinsics(const std::string& extrinsics_path) {
    if (if_load_extrinsics_) {
        return true;
    }

    if (!extrinsics_path.empty()) {
        std::ifstream ifs(extrinsics_path, std::ifstream::in);
        if (!ifs.good()) {
            std::cout << "lidar extrinsics file: " << extrinsics_path << " not exist." << std::endl;
            return false;
        } else {
            YAML::Node ext_file = YAML::LoadFile(extrinsics_path);  //将yaml文件导出到ext_file变量
            LoadLidarQuaternion(ext_file);
            return true;
        }
    }

    // 如果mcap包里存在 lidar extrinsics file
    {
        std::string buffer;
        std::string new_new_extrinsics_file_name = "conf_calib_lidar/roof_lidar_params";
        std::string new_extrinsics_file_name = "lidar_extrinsics.yaml";
        std::string extrinsics_file_name = "/app/runtime_service/neta_lidar/conf/lidar_extrinsics.yaml";
        bool ret = false;
        ret = read_Attachment(new_new_extrinsics_file_name, buffer);
        if (ret) {
            // 持久化读取出来的是json
            std::shared_ptr<YAML::Node> p_yaml = lidarExtrinsicsParaConvert(buffer);
            LoadLidarQuaternion(*p_yaml.get());
            BAG_LOG_INFO << "read lidar correction file in mcap success.";
            return true;
        } else {
            ret = read_Attachment(new_extrinsics_file_name, buffer);
        }
        if (!ret) {
            ret = read_Attachment(extrinsics_file_name, buffer);
        }
        
        if (ret) {
            YAML::Node ext_file = YAML::Load(buffer.c_str());  //将yaml文件导出到ext_file变量
            LoadLidarQuaternion(ext_file);
            BAG_LOG_INFO << "read lidar correction file in mcap success.";
            return true;
        }
    }

    return false;
}

std::string MessageProcess::RePrecision(double num) {
    std::stringstream stream;
    int precision = 8; // 要保留的小数位数
    stream << std::fixed << std::setprecision(precision) << num;
    // printf("before : %4.10f, after : %4.10f\n", num, ans);
    return stream.str();
}

std::shared_ptr<YAML::Node> MessageProcess::lidarExtrinsicsParaConvert(std::string& json) {
    auto yaml = std::make_shared<YAML::Node>();
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    std::istringstream jsonStream(json);
    Json::Reader reader;
    reader.parse(jsonStream, root);
    // 检查解析是否成功
    if (!reader.good()) {
        std::cerr << "Error parsing JSON: " << reader.getFormattedErrorMessages() << std::endl;
        return nullptr;
    }
    (*yaml)["data"] = root["date"].asString();
    (*yaml)["frame_id"] = root["frame_id"].asString();
    (*yaml)["lidar_sn"] = root["lidar_sn"].asString();
    (*yaml)["rotation"]["qw"] =  RePrecision(root["rotation"]["qw"].asDouble());
    (*yaml)["rotation"]["qx"] =  RePrecision(root["rotation"]["qx"].asDouble());
    (*yaml)["rotation"]["qy"] =  RePrecision(root["rotation"]["qy"].asDouble());
    (*yaml)["rotation"]["qz"] =  RePrecision(root["rotation"]["qz"].asDouble());
    (*yaml)["translation"]["x"] =  RePrecision(root["translation"]["x"].asDouble());
    (*yaml)["translation"]["y"] =  RePrecision(root["translation"]["y"].asDouble());
    (*yaml)["translation"]["z"] =  RePrecision(root["translation"]["z"].asDouble());
    (*yaml)["euler_angle"]["yaw"]         =  RePrecision(root["euler_angle"]["yaw"].asDouble());
    (*yaml)["euler_angle"]["pitch"]       =  RePrecision(root["euler_angle"]["pitch"].asDouble());
    (*yaml)["euler_angle"]["roll"]        =  RePrecision(root["euler_angle"]["roll"].asDouble());
    (*yaml)["euler_angle"]["delta_yaw"]   =  RePrecision(root["euler_angle"]["delta_yaw"].asDouble());
    (*yaml)["euler_angle"]["delta_pitch"] =  RePrecision(root["euler_angle"]["delta_pitch"].asDouble());
    return yaml;
}

void MessageProcess::LoadLidarQuaternion(YAML::Node& ext_file) {
    float qw, qx, qy, qz;  //用四元数表示旋转矩阵
    float x, y, z;         //xyz表示平移矢量

    qw = ext_file["rotation"]["qw"].as<float>();
    qx = ext_file["rotation"]["qx"].as<float>();
    qy = ext_file["rotation"]["qy"].as<float>();
    qz = ext_file["rotation"]["qz"].as<float>();
    x = ext_file["translation"]["x"].as<float>();
    y = ext_file["translation"]["y"].as<float>();
    z = ext_file["translation"]["z"].as<float>();

    Eigen::Quaternionf quater;
    quater.x() = qx;
    quater.y() = qy;
    quater.z() = qz;
    quater.w() = qw;

    float yaw, pitch, roll;
    yaw = ext_file["euler_angle"]["yaw"].as<float>();      //偏航角
    pitch = ext_file["euler_angle"]["pitch"].as<float>();  //俯仰角
    roll = ext_file["euler_angle"]["roll"].as<float>();    //横滚角
    yaw = yaw / 180 * M_PI;
    pitch = pitch / 180 * M_PI;
    roll = roll / 180 * M_PI;
    Eigen::Matrix3f R;  //创建旋转矩阵
    R = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

    Eigen::Quaternionf q(R);  //创建四元数

    Eigen::Translation3f translation(x, y, z);  //创建平移变换

    // use euler angle
    lidar_extrinsics = translation * q.toRotationMatrix();
}

template <typename ProtoType>
void MessageProcess::SerializedBagMessage2Proto(const std::shared_ptr<rosbag2_storage::SerializedBagMessage>& inMessage, ProtoType& protoMsg) {
    // inMessage==>rawPointCloud; GenerateMessageByType
    SerializedPayload_t payload;
    payload.reserve(inMessage->serialized_data->buffer_length);
    memcpy(payload.data, inMessage->serialized_data->buffer, inMessage->serialized_data->buffer_length);
    payload.length = inMessage->serialized_data->buffer_length;

    std::string proto_name = protoMsg.GetTypeName();
    CmProtoBuf temp_cmProtoBuf;
    temp_cmProtoBuf.name(proto_name);

    CmProtoBufPubSubType sub_type;
    sub_type.deserialize(&payload, &temp_cmProtoBuf);

    protoMsg.ParseFromArray(temp_cmProtoBuf.str().data(), temp_cmProtoBuf.str().size());
}

void MessageProcess::Proto2BagMessage(const google::protobuf::Message& proto_msg, BagMessage& outMessage) {
    std::string proto_name = proto_msg.GetTypeName();
    CmProtoBuf proto_idl_data;
    proto_idl_data.name(proto_name);
    std::string serialized_data;
    proto_msg.SerializeToString(&serialized_data);
    proto_idl_data.str().assign(serialized_data.begin(), serialized_data.end());

    // SerializedPayload_t payload;
    // CmProtoBufPubSubType sub_type;
    // payload.reserve(sub_type.getSerializedSizeProvider(&proto_idl_data)());
    // sub_type.serialize(&proto_idl_data, &payload);
    // outMessage.message.copy(&payload, false);

    CmProtoBufPubSubType sub_type;
    outMessage.data.m_payload->reserve(sub_type.getSerializedSizeProvider(&proto_idl_data)());
    sub_type.serialize(&proto_idl_data, outMessage.data.m_payload.get());
}

int MessageProcess::Process(const std::shared_ptr<rosbag2_storage::SerializedBagMessage> inMessage, BagMessage& outMessage, const std::string topicName) {

    if (!if_load_correction_) {
        return -1;
    }

    // inMessage==>rawPointCloud;
    hozon::soc::RawPointCloud rawPointCloud;
    SerializedBagMessage2Proto(inMessage, rawPointCloud);

    // process raw_pointCloud ==> pointCloud
    hozon::soc::PointCloud pointCloud;
    int ret = Parse(rawPointCloud, pointCloud);
    if (ret != 0) {
        return -1;
    }

    // pointCloud==>outMessage;
    Proto2BagMessage(pointCloud, outMessage);
    outMessage.time = inMessage->time_stamp;
    outMessage.topic = "/soc/pointcloud";
    outMessage.type = "CmProtoBuf";
    return 0;
}

bool MessageProcess::loadCorrectionFile(const std::string& correction_path) {
    // BAG_LOG_WARN << "load correction file from config ATD128P.dat now:";
    if (if_load_correction_) {
        return true;
    }

    if (!correction_path.empty()) {
        std::ifstream ifs(correction_path, std::ifstream::in);
        if (!ifs.good()) {
            std::cout << "lidar correction file: " << correction_path << " not exist." << std::endl;
            return false;
        }

        if (ifs.is_open()) {
            int length = 0;
            ifs.seekg(0, std::ios::end);  //设置 get 指针的位置，该指针确定下一个读取操作将开始的位置,std::ios::end用于指示应相对于文件末尾设置位置
            length = ifs.tellg();         //使用'seektellg()获取当前位置，该位置表示文件的大小
            ifs.seekg(0, std::ios::beg);  //std::ios::beg用于指示应相对于文件的开头设置位置
            char* buffer = new char[length];
            ifs.read(buffer, length);
            ifs.close();
            int ret = loadCorrectionString(buffer);
            if (ret == 0) {
                return true;
            }
        }

        return false;
    }

    // 如果mcap包里存在correction file
    {
        std::string buffer;
        std::string new_new_cprrection_file_name = "/cfg/lidar_intrinsic_param/ATD128P.dat";
        std::string new_cprrection_file_name = "ATD128P.dat";
        std::string cprrection_file_name = "/app/runtime_service/neta_lidar/conf/ATD128P.dat";
        bool ret = false;

        ret = read_Attachment(new_new_cprrection_file_name, buffer);
        if (!ret) {
            ret = read_Attachment(new_cprrection_file_name, buffer);
        }
        if (!ret) {
            ret = read_Attachment(cprrection_file_name, buffer);
        }
        if (ret) {
            int result = loadCorrectionString(buffer.c_str());
            if (result == 0) {
                BAG_LOG_INFO << "read lidar correction file in mcap success.";
                return true;
            }
        }
    }

    return false;
}

int MessageProcess::loadCorrectionString(const char* data) {
    try {
        const char* p = data;
        PandarATCorrectionsHeader header = *(PandarATCorrectionsHeader*)p;
        if (0xee == header.delimiter[0] && 0xff == header.delimiter[1]) {
            switch (header.version[1]) {
                case 3: {

                } break;
                case 5: {
                    m_PandarAT_corrections.header = header;
                    auto frame_num = m_PandarAT_corrections.header.frame_number;
                    auto channel_num = m_PandarAT_corrections.header.channel_number;
                    p += sizeof(PandarATCorrectionsHeader);
                    memcpy((void*)&m_PandarAT_corrections.start_frame, p, sizeof(uint32_t) * frame_num);
                    p += sizeof(uint32_t) * frame_num;
                    memcpy((void*)&m_PandarAT_corrections.end_frame, p, sizeof(uint32_t) * frame_num);
                    p += sizeof(uint32_t) * frame_num;
                    memcpy((void*)&m_PandarAT_corrections.azimuth, p, sizeof(int32_t) * channel_num);
                    p += sizeof(int32_t) * channel_num;
                    memcpy((void*)&m_PandarAT_corrections.elevation, p, sizeof(int32_t) * channel_num);
                    p += sizeof(int32_t) * channel_num;
                    auto adjust_length = channel_num * 180;
                    memcpy((void*)&m_PandarAT_corrections.azimuth_offset, p, sizeof(int8_t) * adjust_length);
                    p += sizeof(int8_t) * adjust_length;
                    memcpy((void*)&m_PandarAT_corrections.elevation_offset, p, sizeof(int8_t) * adjust_length);
                    p += sizeof(int8_t) * adjust_length;
                    memcpy((void*)&m_PandarAT_corrections.SHA256, p, sizeof(uint8_t) * 32);
                    p += sizeof(uint8_t) * 32;

                    for (int i = 0; i < frame_num; ++i) {
                        m_PandarAT_corrections.start_frame[i] = m_PandarAT_corrections.start_frame[i] * m_PandarAT_corrections.header.resolution;  //resolution代表单位
                        m_PandarAT_corrections.end_frame[i] = m_PandarAT_corrections.end_frame[i] * m_PandarAT_corrections.header.resolution;
                    }

                    for (int i = 0; i < 128; i++) {
                        m_PandarAT_corrections.azimuth[i] = m_PandarAT_corrections.azimuth[i] * m_PandarAT_corrections.header.resolution;
                        m_PandarAT_corrections.elevation[i] = m_PandarAT_corrections.elevation[i] * m_PandarAT_corrections.header.resolution;
                    }
                    for (int i = 0; i < (m_PandarAT_corrections.header.channel_number * 180); i++) {
                        m_PandarAT_corrections.azimuth_offset[i] = m_PandarAT_corrections.azimuth_offset[i] * m_PandarAT_corrections.header.resolution;
                        m_PandarAT_corrections.elevation_offset[i] = m_PandarAT_corrections.elevation_offset[i] * m_PandarAT_corrections.header.resolution;
                    }

                    return 0;
                } break;
                default:
                    break;
            }
        }
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 1;
}

//单位为1/25600度
int MessageProcess::Parse(const hozon::soc::RawPointCloud& rawPointCloud, hozon::soc::PointCloud& pointCloud) {
    if (!if_load_correction_) {
        std::cout << "no correction file, return." << std::endl;
        return -1;
    }

    // 解析原始激光雷达数据
    std::size_t data_size = rawPointCloud.data().size();
    std::size_t frame_size = sizeof(SubPointCloudFrameDEA);
    if (data_size % frame_size != 0) {
        // BAG_LOG_WARN << "rawPointCloud data lenght error!";
        std::cout << "rawPointCloud data lenght error!" << std::endl;
        return -1;
    }
    std::uint32_t frame_num = data_size / frame_size;
    const uint8_t* data_ptr = (uint8_t*)rawPointCloud.data().c_str();
    std::uint32_t pack_num = 0;
    std::uint32_t height = 0;
    for (std::uint32_t k = 0; k < frame_num; k++) {
        SubPointCloudFrameDEA* frame = (SubPointCloudFrameDEA*)(data_ptr + k * frame_size);
        std::uint32_t block_num = frame->data_header.block_num;
        std::uint32_t laser_num = frame->data_header.laser_num;

        // testcode:
        // if (max_point_timestamp < frame->data_tail.timestamp) {
        //     max_point_timestamp = frame->data_tail.timestamp;

        //     if (max_point_timestamp > 999999) {
        //         std::cout << "testlog: frame->data_tail.timestamp: " << frame->data_tail.timestamp << std::endl;
        //         std::cout << "testlog: data_size: " << data_size << std::endl;
        //         std::cout << "testlog: frame_size: " << frame_size << std::endl;
        //         std::cout << "testlog: frame_num: " << frame_num << std::endl;
        //         // 打印原始数据
        //         for (char c : rawPointCloud.data()) {
        //             std::cout << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
        //                     << static_cast<int>(c) << ' ';
        //         }
        //         std::cout << std::endl;
        //     }
        // }
        

        //遍历120°原始点云数据
        for (std::uint32_t i = 0; i < block_num; i++) {
            //计算码盘角度
            uint16_t azimuth = static_cast<uint16_t>(frame->data_body.data_block[i].azimuth);
            uint8_t fine_azimuth = static_cast<uint8_t>(frame->data_body.data_block[i].fine_azimuth);
            int32_t Azimuth = azimuth * 256 + fine_azimuth;

            //获取Hesai的frameid
            int count = 0, field = 0;
            while (count < m_PandarAT_corrections.header.frame_number &&
                   (((Azimuth + MAX_AZI_LEN - m_PandarAT_corrections.start_frame[field]) % MAX_AZI_LEN + (m_PandarAT_corrections.end_frame[field] + MAX_AZI_LEN - Azimuth) % MAX_AZI_LEN) !=
                    (m_PandarAT_corrections.end_frame[field] + MAX_AZI_LEN - m_PandarAT_corrections.start_frame[field]) % MAX_AZI_LEN)) {
                field = (field + 1) % m_PandarAT_corrections.header.frame_number;
                count++;
            }

            if (count >= m_PandarAT_corrections.header.frame_number)
                continue;

            //遍历128线
            for (std::uint32_t j = 0; j < laser_num; j++) {
                auto azimuth = ((Azimuth + MAX_AZI_LEN - m_PandarAT_corrections.start_frame[field]) * 2 - m_PandarAT_corrections.azimuth[j] +
                                m_PandarAT_corrections.getAzimuthAdjustV3(j, Azimuth) * 256);  //单位为：1/25600°
                azimuth = (MAX_AZI_LEN + azimuth) % MAX_AZI_LEN;

                auto elevation = (m_PandarAT_corrections.elevation[j] + m_PandarAT_corrections.getElevationAdjustV3(j, Azimuth) * 256);
                elevation = (MAX_AZI_LEN + elevation) % MAX_AZI_LEN;

                float distance = static_cast<float>(frame->data_body.data_block[i].channel_data[j].distance) * 0.004;  //单位是米
                //去除噪点
                if ((distance < 1.0) || (distance > 210.0)) {
                    continue;
                }
                PointField point;
                point.distance = (double)distance;
                double xyDistance = distance * m_PandarAT_corrections.cos_map[(elevation)];
                point.x = xyDistance * m_PandarAT_corrections.sin_map[(azimuth)];
                point.y = xyDistance * m_PandarAT_corrections.cos_map[(azimuth)];
                point.z = point.distance * m_PandarAT_corrections.sin_map[(elevation)];
                point.time = frame->data_tail.timestamp * 1000;
                point.pitch = (double)elevation;
                point.yaw = (double)azimuth;
                point.intensity = (uint32_t)frame->data_body.data_block[i].channel_data[j].reflectivity;
                point.ring = (uint32_t)j;
                point.block = (uint32_t)pack_num;

                // 保存点云信息到算法的proto中
                auto add = pointCloud.add_points();
                add->set_x(point.x);
                add->set_y(point.y);
                add->set_z(point.z);
                add->set_time(point.time);
                add->set_distance(point.distance);
                add->set_pitch(point.pitch);
                add->set_yaw(point.yaw);
                add->set_intensity(point.intensity);
                add->set_ring(point.ring);
                add->set_block(point.block);
                height++;
            }
            pack_num++;
        }
    }

    // 运动补偿
    if (if_load_extrinsics_) {
        if (rawPointCloud.has_location_data_header() && rawPointCloud.has_location_data_tail()) {
            lidar_start_pose = location_parse((uint8_t*)rawPointCloud.location_data_header().c_str(), rawPointCloud.location_data_header_length());
            lidar_end_pose = location_parse((uint8_t*)rawPointCloud.location_data_tail().c_str(), rawPointCloud.location_data_tail_length());
            motion_compensate(pointCloud, lidar_start_pose, lidar_end_pose, lidar_extrinsics, pack_num);
        }
    }

    pointCloud.set_is_valid(1);
    pointCloud.set_is_big_endian(0);
    pointCloud.set_height(height);
    pointCloud.set_width(1);
    pointCloud.set_point_step(sizeof(PointField));
    pointCloud.set_row_step(height * sizeof(PointField));
    pointCloud.set_is_dense(1);
    pointCloud.set_ecu_serial_number("hesai-AT128P");
    pointCloud.mutable_header()->mutable_sensor_stamp()->set_lidar_stamp(rawPointCloud.header().sensor_stamp().lidar_stamp());
    pointCloud.mutable_header()->set_publish_stamp(rawPointCloud.header().publish_stamp());
    pointCloud.mutable_header()->set_seq(frame_id);
    frame_id++;

    return 0;
}

Eigen::Affine3f MessageProcess::location_parse(const uint8_t* cm_buf, const uint32_t len) {
    std::shared_ptr<hozon::localization::Localization> msg(new hozon::localization::Localization);
    msg->ParseFromArray(cm_buf, len);
    return DrToAffine3f(msg);
}

Eigen::Affine3f MessageProcess::DrToAffine3f(std::shared_ptr<hozon::localization::Localization> proto) {
    Eigen::Quaternionf rotation(proto->pose_local().quaternion().w(), proto->pose_local().quaternion().x(), proto->pose_local().quaternion().y(), proto->pose_local().quaternion().z());
    Eigen::Vector3f position(proto->pose_local().position().x(), proto->pose_local().position().y(), proto->pose_local().position().z());
    // LIDAR_LOG_TRACE << "success position().x() is: "<< position(0) << "  "<<position(1)<< "  "<<position(2)<< "  ";
    Eigen::Matrix3f rotation_matrix = rotation.toRotationMatrix();  // 步骤 1: 创建一个旋转矩阵
    Eigen::Affine3f affine_matrix = Eigen::Affine3f::Identity();    // 步骤 2: 创建一个仿射矩阵并设置旋转部分
    affine_matrix.linear() = rotation_matrix;
    affine_matrix.translation() = position;  // 步骤 3: 设置仿射矩阵的平移部分

    return affine_matrix;
}

void MessageProcess::motion_compensate(hozon::soc::PointCloud& pointCloud, const Eigen::Affine3f& start_pose, const Eigen::Affine3f& end_pose, const Eigen::Affine3f& lidar_extrinsics,
                                       uint32_t pack_num) {
    std::vector<Eigen::Affine3f> motion_poses;
    Eigen::Vector3f translation =  //计算两个位姿之间的平移信息
        start_pose.translation() - end_pose.translation();
    Eigen::Quaternionf q_max(end_pose.linear());  //表示停止姿态的四元数
    Eigen::Quaternionf q_min(start_pose.linear());
    Eigen::Quaternionf q1(q_max.conjugate() * q_min);       //计算两个姿态之间的相对旋转，用四元数表示
    Eigen::Quaternionf q0(Eigen::Quaternionf::Identity());  //单位四元数
    q1.normalize();                                         //对q1进行归一化处理
    translation = q_max.conjugate() * translation;          //将平移矢量应用于最大姿态的共轭旋转
    float d = q0.dot(q1);                                   //计算q0与q1的点积
    float abs_d = abs(d);
    float theta = acos(abs_d);  //theta表示旋转角度
    float sin_theta = sin(theta);
    float c1_sign = (d > 0.0) ? 1.0 : -1.0;  //判断正负

    // int block_num = pointCloud.height();
    for (uint32_t j = 0; j < pack_num; ++j) {                        //按列遍历
        float t = static_cast<float>(pack_num - j) / pack_num;       //列数越大，t越小，t 表示从 0 到 1 的插值参数，可以表示补偿比例
        Eigen::Translation3f ti(t * translation);                    //表示当前阶段 t 对应的平移变换
        float c0 = sin((1.0 - t) * theta) / sin_theta;               //计算插值参数 t 对应的旋转插值系数
        float c1 = sin(t * theta) / sin_theta * c1_sign;             //c0 和 c1 是用于线性插值旋转四元数 q0 和 q1 的系数
        Eigen::Quaternionf qi(c0 * q0.coeffs() + c1 * q1.coeffs());  //qi表示当前阶段 t 对应的旋转四元数
        Eigen::Affine3f trans = Eigen::Affine3f::Identity();         //存储运动补偿后的位姿
        if (abs_d < 1.0 - 1.0e-8) {                                  //根据 abs_d 的值，选择不同的运动补偿方式
            trans = ti * qi * lidar_extrinsics;                      //旋转变化不大时，将平移变换 ti 应用到旋转后的位姿 qi 上
        } else {
            trans = ti * lidar_extrinsics;
        }
        motion_poses.push_back(trans);  //得到的运动补偿后的位姿 trans 被添加到motion_poses向量中
    }

    google::protobuf::RepeatedPtrField<hozon::soc::PointField>* point_vec = pointCloud.mutable_points();
    for (auto itr = point_vec->begin(); itr != point_vec->end(); itr++) {
        auto point = *itr;
        Eigen::Vector3f vec(point.x(), point.y(), point.z());
        vec = motion_poses[point.block()] * vec;
        itr->set_x(vec(0));
        itr->set_y(vec(1));
        itr->set_z(vec(2));
    }
}

bool MessageProcess::read_Attachment(const std::string file_name, std::string& data) {
    if (_reader == nullptr) {
        return false;
    }
    auto attachmentPtr = _reader->read_Attachment(file_name);
    if (attachmentPtr == nullptr) {
        return false;
    }
    data = attachmentPtr->data;
    return true;
}

}  // namespace bag
}  // namespace netaos
}  // namespace hozon
