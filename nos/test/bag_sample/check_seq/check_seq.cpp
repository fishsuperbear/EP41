#include <signal.h>
#include <iostream>
#include "data_tools/bag/reader.h"

#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
// #include "proto/drivers/point_cloud.pb.h"
// #include "proto/drivers/radar.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
// #include "proto/soc/for_test.pb.h"
#include "proto/soc/sensor_image.pb.h"

#include "proto/dreamview/point_cloud.pb.h"
#include "proto/soc/sensor_image.pb.h"  // proto 数据变量

#include <dirent.h>
#include <sys/stat.h>

using namespace hozon::netaos::bag;

// std::vector<std::string> check_topic = {"/soc/pointcloud"};

bool isdir(std::string& path) {
    struct stat fileStat;
    if (stat(path.c_str(), &fileStat) == 0) {
        if (S_ISREG(fileStat.st_mode)) {
            printf("这是一个文件\n");
            return false;
        } else if (S_ISDIR(fileStat.st_mode)) {
            printf("这是一个目录\n");
            return true;
        } else {
            printf("既不是文件也不是目录\n");
            return false;
        }
    } else {
        printf("获取文件/目录信息失败\n");
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc == 3) {
        std::string dirpath = argv[1];
        std::vector<std::string> check_file;

        if (isdir(dirpath)) {
            DIR* dir;
            struct dirent* entry;
            struct stat fileStat;

            // 打开目录
            dir = opendir(dirpath.c_str());
            if (dir == NULL) {
                perror("无法打开目录");
                exit(EXIT_FAILURE);
            }

            // 遍历目录
            while ((entry = readdir(dir)) != NULL) {
                char full_path[PATH_MAX];
                snprintf(full_path, PATH_MAX, "%s/%s", dirpath.c_str(), entry->d_name);

                // 获取文件信息
                if (stat(full_path, &fileStat) == 0) {
                    // 检查文件是否以".map"结尾，并且是普通文件
                    if (S_ISREG(fileStat.st_mode) && strstr(entry->d_name, ".map") != NULL) {
                        printf("找到以.map结尾的文件: %s\n", entry->d_name);
                        check_file.push_back(dirpath + std::string(entry->d_name));
                    }
                } else {
                    perror("获取文件信息失败");
                }
            }

            // 关闭目录
            closedir(dir);

        } else {
            check_file.push_back(dirpath);
        }

        std::vector<std::string> check_topic;

        check_topic.push_back(argv[2]);
        for (auto topic_name : check_topic) {
            for (auto file_path : check_file) {
                std::cout << "topic_name: " << topic_name << std::endl;
                Reader reader;
                reader.Open(file_path, "mcap");  //path为.mcap包路经
                std::cout << "open: " << file_path << std::endl;

                auto topics_info = reader.GetAllTopicsAndTypes();  //获取包中所有的topic和cm type
                // auto it = std::find(topics_info.begin(), topics_info.end(), topic_name);
                // 判断是否找到
                if (topics_info.find(topic_name) != topics_info.end()) {
                    //有topic
                    std::vector<std::string> topics;
                    topics.push_back(topic_name);
                    reader.SetFilter(topics);  //设置过滤器，只读取指定的topic的message
                    int last_id = -1;

                    //直接取出一个message并返回id
                    while (reader.HasNext()) {
                        int this_id = -1;
                        if (std::string::npos != topic_name.find("/soc/encoded_camera")) {
                            hozon::soc::CompressedImage proto_data = reader.ReadNextProtoMessage<hozon::soc::CompressedImage>();
                            this_id = proto_data.mutable_header()->seq();
                        } else if ("/soc/pointcloud" == topic_name) {
                            hozon::soc::PointCloud proto_data = reader.ReadNextProtoMessage<hozon::soc::PointCloud>();
                            this_id = proto_data.mutable_header()->seq();
                        }

                        if (this_id >= 0) {
                            if (-1 == last_id) {
                                last_id = this_id;
                                std::cout << "begin id = " << last_id << std::endl;
                            } else {
                                if ((last_id + 1) != this_id) {
                                    std::cout << "pause: pre_id=" << last_id << ", next_id=" << this_id << std::endl;
                                    last_id = -1;
                                } else {
                                    last_id = this_id;
                                }
                            }
                        } else {
                            std::cout << "get seq id error" << std::endl;
                        }
                    }
                    std::cout << "end id=" << last_id << std::endl;

                } else {
                    std::cout << "no topic" << std::endl;
                }
            }
        }
    }
    return 0;
}