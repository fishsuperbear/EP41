#include <unistd.h>
#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/SubscriberAttributes.h>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>

#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/SubscriberAttributes.h>

#include <fastrtps/types/DynamicDataHelper.hpp>
#include <fastrtps/types/DynamicDataFactory.h>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/subscriber/SampleInfo.h>
#include <fastrtps/rtps/common/Types.h>

#include <fastrtps/types/TypeIdentifier.h>
#include <fastrtps/types/TypeObject.h>
#include <fastrtps/types/TypeObjectFactory.h>

#include <fastrtps/attributes/SubscriberAttributes.h>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/dds/subscriber/SampleInfo.hpp>

using namespace eprosima::fastdds::dds;
using eprosima::fastrtps::types::ReturnCode_t;

std::vector<std::string> default_network_List = {
    "127.0.0.1",    // x86 loopback 
    "192.168.10.6"  // MDC
};

class HelloWorldSubscriber
{
public:
    HelloWorldSubscriber();
    virtual ~HelloWorldSubscriber();

    bool init();
    void run();
    void deinit();

private:
    eprosima::fastdds::dds::DomainParticipant* mp_participant;
    eprosima::fastdds::dds::Subscriber* mp_subscriber;
    std::map<eprosima::fastdds::dds::DataReader*, eprosima::fastdds::dds::Topic*> topics_;
    std::map<eprosima::fastdds::dds::DataReader*, eprosima::fastrtps::types::DynamicType_ptr> readers_;
    std::map<eprosima::fastdds::dds::DataReader*, eprosima::fastrtps::types::DynamicData_ptr> datas_;
    eprosima::fastrtps::SubscriberAttributes att_;
    eprosima::fastdds::dds::DataReaderQos qos_;

    bool _stop_flag = false;

public:
    class SubListener
        :  public eprosima::fastdds::dds::DomainParticipantListener
    {
    public:
        SubListener(
                HelloWorldSubscriber* sub)
            : n_matched(0)
            , n_samples(0)
            , subscriber_(sub)
        {
        }
        ~SubListener() override
        {
        }
        void on_data_available(
                eprosima::fastdds::dds::DataReader* reader) override;
        void on_subscription_matched(
                eprosima::fastdds::dds::DataReader* reader,
                const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;
        void on_type_information_received(
                eprosima::fastdds::dds::DomainParticipant* participant,
                const eprosima::fastrtps::string_255 topic_name,
                const eprosima::fastrtps::string_255 type_name,
                const eprosima::fastrtps::types::TypeInformation& type_information) override;
        int n_matched;
        uint32_t n_samples;
        HelloWorldSubscriber* subscriber_;
    }
    m_listener;

};

HelloWorldSubscriber::HelloWorldSubscriber()
    : mp_participant(nullptr)
    , mp_subscriber(nullptr)
    , m_listener(this)
{
}

bool HelloWorldSubscriber::init()
{
    DomainParticipantQos particpant_qos = PARTICIPANT_QOS_DEFAULT;
    particpant_qos.transport().use_builtin_transports = false;
    auto udp_transport = std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();
    udp_transport->interfaceWhiteList = default_network_List;
    particpant_qos.transport().user_transports.push_back(udp_transport);

    particpant_qos.wire_protocol().builtin.typelookup_config.use_client = true;

    //Do not enable entities on creation
    DomainParticipantFactoryQos factory_qos;
    factory_qos.entity_factory().autoenable_created_entities = false;
    DomainParticipantFactory::get_instance()->set_qos(factory_qos);

    StatusMask par_mask = StatusMask::subscription_matched() << StatusMask::data_available();
    mp_participant = DomainParticipantFactory::get_instance()->create_participant(0, particpant_qos, &m_listener, par_mask);
    if (mp_participant == nullptr)
    {
        return false;
    }
    if (mp_participant->enable() != ReturnCode_t::RETCODE_OK)
    {
        DomainParticipantFactory::get_instance()->delete_participant(mp_participant);
        return false;
    }

    return true;
}

HelloWorldSubscriber::~HelloWorldSubscriber()
{
    for (const auto& it : topics_)
    {
        mp_subscriber->delete_datareader(it.first);
        mp_participant->delete_topic(it.second);
    }
    if (mp_subscriber != nullptr)
    {
        mp_participant->delete_subscriber(mp_subscriber);
    }

    DomainParticipantFactory::get_instance()->delete_participant(mp_participant);
    topics_.clear();
    readers_.clear();
    datas_.clear();
}

void HelloWorldSubscriber::SubListener::on_subscription_matched(
        DataReader*,
        const SubscriptionMatchedStatus& info)
{
    if (info.current_count_change == 1)
    {
        n_matched = info.total_count;
        std::cout << "Subscriber matched" << std::endl;
    }
    else if (info.current_count_change == -1)
    {
        n_matched = info.total_count;
        std::cout << "Subscriber unmatched" << std::endl;
    }
    else
    {
        std::cout << info.current_count_change
                  << " is not a valid value for SubscriptionMatchedStatus current count change" << std::endl;
    }
}

void HelloWorldSubscriber::SubListener::on_data_available(
        DataReader* reader)
{
    std::cout << "on_data_available ......" << std::endl;
    auto dit = subscriber_->datas_.find(reader);

    if (dit != subscriber_->datas_.end())
    {
        eprosima::fastrtps::types::DynamicData_ptr data = dit->second;
        SampleInfo info;
        if (reader->take_next_sample(data.get(), &info) == ReturnCode_t::RETCODE_OK)
        {
            if (info.instance_state == ALIVE_INSTANCE_STATE)
            {
                eprosima::fastrtps::types::DynamicType_ptr type = subscriber_->readers_[reader];
                this->n_samples++;
                std::cout << "Received data of type " << type->get_name() << std::endl;
                //eprosima::fastrtps::types::DynamicDataHelper::print(data);
            }
        }
    }
}

void HelloWorldSubscriber::SubListener::on_type_information_received(
        eprosima::fastdds::dds::DomainParticipant* participant,
        const eprosima::fastrtps::string_255 topic_name,
        const eprosima::fastrtps::string_255 type_name,
        const eprosima::fastrtps::types::TypeInformation& type_information) 
{
    std::cout << "on_type_information_received topic " << topic_name << " type_name : " << type_name << std::endl;
    std::function<void(const std::string&, const eprosima::fastrtps::types::DynamicType_ptr)> callback =
            [this, topic_name](const std::string& name, const eprosima::fastrtps::types::DynamicType_ptr type)
            {
                std::cout << "Discovered type: " << name << " from topic " << topic_name << std::endl;

                //CREATE THE SUBSCRIBER
                if (subscriber_->mp_subscriber == nullptr)
                {
                    subscriber_->mp_subscriber = subscriber_->mp_participant->create_subscriber(
                        SUBSCRIBER_QOS_DEFAULT, nullptr);
                    if (subscriber_->mp_subscriber == nullptr)
                    {
                        return;
                    }
                }

                //CREATE THE TOPIC
                eprosima::fastdds::dds::Topic* topic = subscriber_->mp_participant->create_topic(
                    static_cast<std::string>(topic_name),
                    name,
                    TOPIC_QOS_DEFAULT);
                if (topic == nullptr)
                {
                    return;
                }

                StatusMask sub_mask = StatusMask::subscription_matched() << StatusMask::data_available();
                DataReader* reader = subscriber_->mp_subscriber->create_datareader(
                    topic,
                    subscriber_->qos_,
                    &subscriber_->m_listener,
                    sub_mask);
                if (type == nullptr)
                {
                    const eprosima::fastrtps::types::TypeIdentifier* ident =
                            eprosima::fastrtps::types::TypeObjectFactory::get_instance()->get_type_identifier_trying_complete(name);

                    if (nullptr != ident)
                    {
                        const eprosima::fastrtps::types::TypeObject* obj =
                                eprosima::fastrtps::types::TypeObjectFactory::get_instance()->get_type_object(ident);

                        eprosima::fastrtps::types::DynamicType_ptr dyn_type =
                                eprosima::fastrtps::types::TypeObjectFactory::get_instance()->build_dynamic_type(name, ident, obj);

                        if (nullptr != dyn_type)
                        {
                            subscriber_->readers_[reader] = dyn_type;
                            eprosima::fastrtps::types::DynamicData_ptr data(
                                eprosima::fastrtps::types::DynamicDataFactory::get_instance()->create_data(dyn_type));
                            subscriber_->datas_[reader] = data;
                        }
                        else
                        {
                            std::cout << "ERROR: DynamicType cannot be created for type: " << name << std::endl;
                        }
                    }
                    else
                    {
                        std::cout << "ERROR: TypeIdentifier cannot be retrieved for type: " << name << std::endl;
                    }
                }
                else
                {
                    subscriber_->topics_[reader] = topic;
                    subscriber_->readers_[reader] = type;
                    eprosima::fastrtps::types::DynamicData_ptr data(eprosima::fastrtps::types::DynamicDataFactory::get_instance()->create_data(type));
                    subscriber_->datas_[reader] = data;
                }
            };

    subscriber_->mp_participant->register_remote_type(
        type_information,
        type_name.to_string(),
        callback);
}

void HelloWorldSubscriber::run()
{
    std::cout << "Subscriber running. Please press enter to stop the Subscriber" << std::endl;
    while(!_stop_flag) {
        sleep(1);
    }
}

void HelloWorldSubscriber::deinit()
{
    std::cout << "Subscriber stop." << std::endl;
    _stop_flag = true;
}
