
#include "phm/include/phm_client.h"
#include "phm/impl/include/phm_client_impl.h"


namespace hozon {
namespace netaos {
namespace phm {


PHMClient::PHMClient()
{
    phm_impl_ = std::make_unique<PHMClientImpl>();
}

PHMClient::~PHMClient()
{
}

int32_t
PHMClient::Init(const std::string phmConfigPath,
                std::function<void(bool)> service_available_callback,
                std::function<void(ReceiveFault_t)> fault_receive_callback,
                const std::string processName)
{
    return phm_impl_->Init(phmConfigPath, service_available_callback, fault_receive_callback, processName);
}

int32_t
PHMClient::Start(uint32_t delayTime)
{
    return phm_impl_->Start(delayTime);
}

void
PHMClient::Stop()
{
    phm_impl_->Stop();
}

void
PHMClient::Deinit()
{
    phm_impl_->Deinit();
}

int32_t
PHMClient::ReportCheckPoint(const uint32_t checkPointId)
{
    phm_impl_->ReportCheckPoint(checkPointId);
    return 0;
}

int32_t
PHMClient::ReportFault(const SendFault_t& faultInfo)
{
    phm_impl_->ReportFault(faultInfo);
    return 0;
}

void
PHMClient::InhibitFault(const std::vector<uint32_t>& faultKeys)
{
    phm_impl_->InhibitFault(faultKeys);
}


void
PHMClient::RecoverInhibitFault(const std::vector<uint32_t>& faultKeys)
{
    phm_impl_->RecoverInhibitFault(faultKeys);
}

void
PHMClient::InhibitAllFault()
{
    phm_impl_->InhibitAllFault();
}

void
PHMClient::RecoverInhibitAllFault()
{
    phm_impl_->RecoverInhibitAllFault();
}

int32_t
PHMClient::GetDataCollectionFile(std::vector<std::string>& outResult)
{
    return 0;
}

int32_t
PHMClient::GetDataCollectionFile(std::function<void(std::vector<std::string>&)> collectionFileCb)
{
    phm_impl_->GetDataCollectionFile(collectionFileCb);
    return 0;
}

}  // namespace phm
}  // namespace netaos
}  // namespace hozon
/* EOF */
