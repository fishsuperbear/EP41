#include "dds_helper.h"

DDSHelper& DDSHelper::GetInstance() {
    static DDSHelper _instance;

    return _instance;
}

DDSHelper::DDSHelper() {
    eprosima::fastdds::dds::DomainParticipantQos participant_qos;
    participant_qos.transport().send_socket_buffer_size = 10 * 1024 * 1024;
    participant_qos.transport().listen_socket_buffer_size = 10 * 1024 * 1024;

    // create participant, domain id is 0, default qos
    _participant = eprosima::fastdds::dds::DomainParticipantFactory::get_instance()->create_participant(
                            0, 
                            participant_qos);
    
    // create subscriber, default qos
    _subscriber = _participant->create_subscriber(
                            eprosima::fastdds::dds::SUBSCRIBER_QOS_DEFAULT, 
                            nullptr);

    // create publisher, default qos                        
    _publisher = _participant->create_publisher(
                            eprosima::fastdds::dds::PUBLISHER_QOS_DEFAULT, 
                            nullptr);
}