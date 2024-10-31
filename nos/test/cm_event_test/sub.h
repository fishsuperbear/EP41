#pragma once 

#include <iostream>
#include <memory>
#include <unistd.h>

#include "cm/include/proxy.h"
#include "idl/generated/avmPubSubTypes.h"
#include "log/include/default_logger.h"

using namespace hozon::netaos::cm;

class sub
{
public:
    sub();
    ~sub();
    void init(const std::string& topic);
    void deinit();
    void run(std::function<void(void)> callback);
    void sub_callback(void);
private:
    std::shared_ptr<avmPubSubType> pubsubtype_;
    std::shared_ptr<Proxy> proxy_;

    std::shared_ptr<avm> data;
    bool _stop_flag = false;
    int count = 0;
    bool print_flag = false;
};

void sub::init(const std::string& topic) {
    pubsubtype_ = std::make_shared<avmPubSubType>();
    proxy_ = std::make_shared<Proxy>(pubsubtype_);
    proxy_->Init(0, topic);

    data = std::make_shared<avm>();
}

void sub::deinit() {
    _stop_flag = true;
    proxy_->Deinit();
}

void sub::run(std::function<void(void)> callback) {
    proxy_->Listen(callback);

    while (!_stop_flag)
    {   
        count++;
        if(count == 3 && print_flag == false ){
            std::cout<<"!!!!!hz_test_failed!!!!!"<<std::endl;
            _stop_flag = true;
            return;
        } 
        sleep(1);
    }
}

void sub::sub_callback() {
    if (proxy_->IsMatched()) {
        proxy_->Take(data);
        DF_LOG_INFO << " Recv data.name : " << data->name() << " data.seq : " << data->header().seq();
        if(data->header().seq() >= 1){
            if(print_flag){
                _stop_flag = true;
                return;
            }
            else{
                std::cout<<"!!!!!hz_test_success!!!!!"<<std::endl;
                print_flag = true;
            }
        }
    }

}

sub::sub()
{
}

sub::~sub()
{
}

