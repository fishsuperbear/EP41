#include <iostream>
#include <ostream>
#include <thread>
#include "map_manage.h"

using namespace hozon::netaos;
using namespace perception_map::semanticmap;

// const std::string map_position = "/home/zz/Documents/ntp_map/";
const std::string map_position = "/opt/usr/hz_map/ntp/";

void writeThread() {
    MapManage& manager = MapManage::getInstance(map_position);
    auto ptr1 = manager.getMapPlanning(11);
    auto ptr2 = manager.getMapSlam(11);
    auto ptr3 = manager.getFeatureMap(11);

    MapManage::Map map;
    map.map_planning = *ptr1;
    map.map_slam = *ptr2;
    map.feature_map = *ptr3;

    for (uint32_t i = 1; i < 10; i++) {
        int ret = manager.saveMap(map);
        if (ret == -1) {
            std::cout << "---------saveMap fail" << std::endl;
        }
        std::cout << "---------writeThread  id = " << ret << std::endl;
    }
}

void readThread() {
    MapManage& manager = MapManage::getInstance(map_position);
    for (uint32_t i = 0; i < 10; i++) {
        std::cout << "---------readThread  id = " << i << std::endl;
        auto ptr = manager.getMapPlanning(i);
        double longitude = ptr->map.header().j02longitude();
        double latitude = ptr->map.header().j02latitude();
        std::string create_time = ptr->map.header().create_time();
        std::cout << "---------create_time = " << create_time << std::endl;
        std::cout << "---------longitude = " << longitude << std::endl;
        std::cout << "---------latitude = " << latitude << std::endl;
    }
}

int main() {

    int ret{};

    MapManage& manager = MapManage::getInstance(map_position);

    ret = manager.setMapId(11);

    // test getMapPlanning
    if (ret < 0 ) {
        std::cout << "setMapId fail" << std::endl;
    }

    ret = manager.getMapId();
    std::cout << "getMapId id = " << ret << std::endl;

    auto ptr1 = manager.getMapPlanning();
    auto ptr2 = manager.getMapSlam();
    auto ptr3 = manager.getFeatureMap();

    if (ptr1 == nullptr || ptr2 == nullptr || ptr3 == nullptr) {
        return -1;
    }

    std::cout << "id = "<< ptr1->id << std::endl;
    std::cout << ptr1->map.DebugString(); // 测试message基类中的方法
    std::cout << ptr1->path.DebugString();
    
    double longitude1 = ptr1->map.header().j02longitude();
    double latitude1 = ptr1->map.header().j02latitude();
    std::string create_time1 = ptr1->map.header().create_time();

    std::cout << "---------create_time1 = " << create_time1 << std::endl;
    std::cout << "---------longitude1 = " << longitude1 << std::endl;
    std::cout << "---------latitude1 = " << latitude1 << std::endl;

    double longitude2 = ptr2->map.header().j02longitude();
    double latitude2 = ptr2->map.header().j02latitude();
    std::string create_time2 = ptr2->map.header().create_time();

    std::cout << "---------create_time2 = " << create_time2 << std::endl;
    std::cout << "---------longitude2 = " << longitude2 << std::endl;
    std::cout << "---------latitude2 = " << latitude2 << std::endl;

    std::cout << "---------FeatureMap Size = " << ptr3->size() << std::endl;

 
    // test save
    {
        MapManage::Map map;
        map.map_planning = *ptr1;
        map.map_slam = *ptr2;
        map.feature_map = *ptr3;

        for (size_t i = 1; i < 10; i++) {
            ret = manager.saveMap(map);
            if (ret == -1) {
                std::cout << "---------saveMap fail"  << std::endl;
                return -1;
            }
            std::cout << "---------saveMap id = " << ret << std::endl;
        }
    }

    // test pollAllMap
    {
        std::vector<uint32_t> vec = manager.pollAllMap();
        for (int i = 0; i < vec.size(); i++) {
            std::cout << "---------pollAllMap vec[" << i << "] = " << vec[i] << "\n";
        }
    }

    // test getXY
    {
        auto xy = manager.getXY(1);
        auto lon = xy.first;
        auto lat = xy.second;
        std::cout << "---------longitude = " << lon << std::endl;
        std::cout << "---------latitude = " << lat << std::endl;
    }

    // test deleteMap
    {
        for (uint32_t i = 6; i <= 9; i++) {
            ret = manager.deleteMap(i);
            if (ret == -1) {
                std::cout << "---------deleteMap fail"  << std::endl;
                return -1;
            }
            std::cout << "---------deleteMap id = " << i << std::endl;
        }
    }

    // test updateMap
    {
        MapManage::Map map;
        map.map_planning = *ptr1;
        map.map_slam = *ptr2;
        map.feature_map = *ptr3;
        auto header = map.map_planning.map.mutable_header();
        header->set_j02longitude(66.66);
        header->set_j02latitude(77.77);

        ret = manager.updateMap(11, map);
        std::cout << "---------updateMap ret = " << ret << std::endl;

        manager.getXY(11);
    }
/* 
    // test multi thread
    {
        std::thread writer(writeThread);
        
        std::thread reader1(readThread);
        std::thread reader2(readThread);
        writer.join();
        reader1.join();
        reader2.join();
    }

    // test multi process
     */ 
}