/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include "ne_someip_log.h"
#include "ne_someip_e2e_result.h"
#include "extend/e2exf/config.h"
#include "extend/e2exf/transformer.h"
#include "extend/e2exf/e2e_handler.h"
#include "config_reader.h"

using e2exf::ConfigurationFormat;
using e2exf::DataIdentifier;
using e2e::Result;

namespace {

class E2EHandlerImpl {
   public:
    E2EHandlerImpl( const E2EHandlerImpl& ) = delete;
    E2EHandlerImpl( E2EHandlerImpl&& )      = delete;

    E2EHandlerImpl& operator=( const E2EHandlerImpl& ) = delete;
    E2EHandlerImpl& operator=( E2EHandlerImpl&& ) = delete;

    static E2EHandlerImpl& instance();

    bool Configure( const std::string& bindingConfigurationPath,
                    ConfigurationFormat      bindingConfigurationFormat,
                    const std::string& e2exfConfigurationPath,
                    ConfigurationFormat      e2exfConfigurationFormat );

    e2e::Result HandleCheckStatus(
        std::uint16_t serviceId, std::uint16_t instanceId, std::uint16_t eventId,
        E2E_state_machine::E2ECheckStatus checkStatus );

    bool ProtectEvent( const std::uint16_t serviceId, const std::uint16_t instanceId,
                       const std::uint16_t eventId, const crc::Buffer& inputBuffer,
                       crc::Buffer& outputBuffer );

    Result CheckEvent( const std::uint16_t serviceId, const std::uint16_t instanceId,
                       const std::uint16_t eventId, const crc::Buffer& inputBuffer,
                       crc::Buffer& outputBuffer );

    bool ProtectMethod( const std::uint16_t serviceId, const std::uint16_t instanceId,
                        const std::uint16_t methodId, const crc::Buffer& inputBuffer,
                        crc::Buffer& outputBuffer );

    Result CheckMethod( const std::uint16_t serviceId, const std::uint16_t instanceId,
                        const std::uint16_t methodId, const crc::Buffer& inputBuffer,
                        crc::Buffer& outputBuffer );

    bool IsProtected( const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventOrMethodId );

    void GetDataIdAndCounter( const std::uint16_t          serviceId,
                              const std::uint16_t          instanceId,
                              const std::uint16_t          eventId,
                              const crc::Buffer& inputBuffer,
                              DataIdentifier&              dataId,
                              std::uint32_t&               counter );

   private:
    bool configured_ = false;
    e2exf::Config config_;
    std::shared_ptr<e2exf::Transformer> transformer_;
    std::map<std::tuple<std::uint16_t, std::uint16_t, std::uint16_t>, DataIdentifier>
        mappingToDataId_;
    std::map<DataIdentifier, std::shared_ptr<E2E_state_machine::StateMachine>>
                      stateMachines_;

    E2EHandlerImpl();
};

E2EHandlerImpl& E2EHandlerImpl::instance() {
    static E2EHandlerImpl theInstance;
    return theInstance;
}

E2EHandlerImpl::E2EHandlerImpl() {}

bool
E2EHandlerImpl::Configure( const std::string&       bindingConfigurationPath,
                           ConfigurationFormat      bindingConfigurationFormat,
                           const std::string&       e2exfConfigurationPath,
                           ConfigurationFormat      e2exfConfigurationFormat ) {
    if ( configured_ ) {
        ne_someip_log_debug("Status handler was already configured, skipping");
        return false;
    }

    try {
        ne_someip_log_debug("loading e2e config from: %s", e2exfConfigurationPath.c_str());
        bool ret = false;
        config_ = LoadE2EConfiguration( e2exfConfigurationPath, e2exfConfigurationFormat, ret );
        if (!ret) {
            ne_someip_log_debug("LoadE2EConfiguration error.");
            return false;
        }
        ne_someip_log_debug("loading e2e config success");

        // dataid mapping
        ne_someip_log_debug("loading dataid mapping from: %s", bindingConfigurationPath.c_str());
        mappingToDataId_ =
            LoadE2EDataIdMapping( bindingConfigurationPath, bindingConfigurationFormat, ret );
        if (!ret || 0 == mappingToDataId_.size()) {
            ne_someip_log_debug("LoadE2EDataIdMapping error.");
            return false;
        }
        ne_someip_log_debug("loading dataid mapping success, size = %d", mappingToDataId_.size());

        // state machines mapping
        ne_someip_log_debug("loading state machines from: %s", e2exfConfigurationPath.c_str());
        stateMachines_ = LoadE2EStateMachines( e2exfConfigurationPath, e2exfConfigurationFormat, ret );
        if (!ret || 0 == stateMachines_.size()) {
            ne_someip_log_debug("LoadE2EStateMachines error.");
            return false;
        }
        ne_someip_log_debug("loading state machines success, size = %d", stateMachines_.size());
    } catch ( std::exception& e ) {
        ne_someip_log_debug("Failed to configure status handler due to: %s", e.what());
        return false;
    } catch ( ... ) {
        ne_someip_log_debug("Failed to configure status handler");
        return false;
    }

    transformer_ = std::make_shared<e2exf::Transformer>( std::move( config_ ) );
    if ( nullptr == transformer_ ) {
        ne_someip_log_error("Failed to make transformer ptr.");
        return false;
    }

    configured_ = true;
    return true;
}

e2e::Result
E2EHandlerImpl::HandleCheckStatus( std::uint16_t serviceId,
                                   std::uint16_t instanceId,
                                   std::uint16_t eventId,
                                   E2E_state_machine::E2ECheckStatus checkStatus ) {
    ne_someip_log_debug("Handle status for serviceId [%d], instanceId [%d], eventId [%d], status [%d]",
        serviceId, instanceId, eventId, static_cast<std::uint16_t>( checkStatus ));

    if ( !configured_ ) {
        ne_someip_log_debug("Status handler was not configured, skipping real check.");
        return Result();
    }

    // it needs to be specified and clarified which result we should return if data is unprotected
    auto mapping = mappingToDataId_.find( std::make_tuple( serviceId, instanceId, eventId ) );
    if ( mapping == mappingToDataId_.end() ) {
        ne_someip_log_debug("No dataid mapping found, skipping real check.");
        return Result();
    }

    auto stateMachine = stateMachines_.find( mapping->second );
    if ( stateMachine == stateMachines_.end() ) {
        ne_someip_log_debug("No state machine mapping found, skipping real check.");
        return Result();
    }

    E2E_state_machine::E2EState state;
    stateMachine->second->Check( checkStatus, state );

    ne_someip_log_debug("Result state = %d", static_cast<std::uint16_t>( state ));

    return Result{state, checkStatus};
}

bool
E2EHandlerImpl::ProtectEvent( const std::uint16_t          serviceId,
                              const std::uint16_t          instanceId,
                              const std::uint16_t          eventId,
                              const crc::Buffer& inputBuffer,
                              crc::Buffer&       outputBuffer ) {
    ne_someip_log_debug("serviceId [%d], instanceId [%d], eventId [%d]", serviceId, instanceId, eventId);
    if ( !configured_ ) {
        ne_someip_log_debug("Status handler was not configured, skipping real check.");
        return false;
    }

    // it needs to be specified and clarified which result we should return if data is unprotected
    auto mapping = mappingToDataId_.find( std::make_tuple( serviceId, instanceId, eventId ) );
    if ( mapping == mappingToDataId_.end() ) {
        ne_someip_log_debug("No dataid mapping found, skipping real check.");
        return false;
    }

    transformer_->ProtectOutOfPlace( mapping->second, inputBuffer, outputBuffer );

    return true;
}

Result
E2EHandlerImpl::CheckEvent( const std::uint16_t          serviceId,
                            const std::uint16_t          instanceId,
                            const std::uint16_t          eventId,
                            const crc::Buffer& inputBuffer,
                            crc::Buffer&       outputBuffer ) {
    ne_someip_log_debug("serviceId [%d], instanceId [%d], eventId [%d]", serviceId, instanceId, eventId);
    if ( !configured_ ) {
        ne_someip_log_debug("Status handler was not configured, skipping real check.");
        return Result();
    }

    // it needs to be specified and clarified which result we should return if data is unprotected
    auto mapping = mappingToDataId_.find( std::make_tuple( serviceId, instanceId, eventId ) );
    if ( mapping == mappingToDataId_.end() ) {
        ne_someip_log_debug("No dataid mapping found, skipping real check.");
        return Result();
    }

    return transformer_->CheckOutOfPlace( mapping->second, inputBuffer, outputBuffer );
}

bool
E2EHandlerImpl::ProtectMethod( const std::uint16_t          serviceId,
                               const std::uint16_t          instanceId,
                               const std::uint16_t          methodId,
                               const crc::Buffer& inputBuffer,
                               crc::Buffer&       outputBuffer ) {
    ne_someip_log_debug("serviceId [%d], instanceId [%d], methodId [%d]", serviceId, instanceId, methodId);
    if ( !configured_ ) {
        ne_someip_log_debug("Status handler was not configured, skipping real check.");
        return false;
    }

    // it needs to be specified and clarified which result we should return if data is unprotected
    auto mapping = mappingToDataId_.find( std::make_tuple( serviceId, instanceId, methodId ) );
    if ( mapping == mappingToDataId_.end() ) {
        ne_someip_log_debug("No dataid mapping found, skipping real check.");
        return false;
    }

    transformer_->ProtectOutOfPlace( mapping->second, inputBuffer, outputBuffer );

    return true;
}

Result
E2EHandlerImpl::CheckMethod( const std::uint16_t          serviceId,
                             const std::uint16_t          instanceId,
                             const std::uint16_t          methodId,
                             const crc::Buffer& inputBuffer,
                             crc::Buffer&       outputBuffer ) {
    ne_someip_log_debug("serviceId [%d], instanceId [%d], methodId [%d]", serviceId, instanceId, methodId);
    if ( !configured_ ) {
        ne_someip_log_debug("Status handler was not configured, skipping real check.");
        return Result();
    }

    // it needs to be specified and clarified which result we should return if data is unprotected
    auto mapping = mappingToDataId_.find( std::make_tuple( serviceId, instanceId, methodId ) );
    if ( mapping == mappingToDataId_.end() ) {
        ne_someip_log_debug("No dataid mapping found, skipping real check.");
        return Result();
    }

    return transformer_->CheckOutOfPlace( mapping->second, inputBuffer, outputBuffer );
}

bool
E2EHandlerImpl::IsProtected( const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventOrMethodId ) {
    if ( !configured_ ) {
        ne_someip_log_debug("Status handler was not configured, skipping real check.");
        return false;
    }

    auto mapping = mappingToDataId_.find( std::make_tuple( serviceId, instanceId, eventOrMethodId ) );
    if ( mapping == mappingToDataId_.end() ) {
        ne_someip_log_debug("No dataid mapping found, skipping real check.");
        return false;
    }

    return transformer_->IsProtected(mapping->second);
}

void
E2EHandlerImpl::GetDataIdAndCounter( const std::uint16_t          serviceId,
                                     const std::uint16_t          instanceId,
                                     const std::uint16_t          eventId,
                                     const crc::Buffer& inputBuffer,
                                     DataIdentifier&              dataId,
                                     std::uint32_t&               counter ) {
    ne_someip_log_debug("serviceId [%d], instanceId [%d], eventId [%d]", serviceId, instanceId, eventId);
    if ( !configured_ ) {
        ne_someip_log_debug("Status handler was not configured, skipping real check.");
        return;
    }

    // it needs to be specified and clarified which result we should return if data is unprotected
    auto mapping = mappingToDataId_.find( std::make_tuple( serviceId, instanceId, eventId ) );
    if ( mapping == mappingToDataId_.end() ) {
        ne_someip_log_debug("No dataid mapping found, skipping real check.");
        return;
    }

    dataId = mapping->second;

    bool ret = transformer_->GetCounter( mapping->second, inputBuffer, counter );
    if (!ret) {
        ne_someip_log_debug("GetCounter error.");
        return;
    }
}

}  // namespace

namespace e2exf {

bool
E2EHandler::Configure( const std::string&       bindingConfigurationPath,
                       ConfigurationFormat      bindingConfigurationFormat,
                       const std::string&       e2exfConfigurationPath,
                       ConfigurationFormat      e2exfConfigurationFormat ) {
    return E2EHandlerImpl::instance().Configure( bindingConfigurationPath,
                                                 bindingConfigurationFormat,
                                                 e2exfConfigurationPath,
                                                 e2exfConfigurationFormat );
}

Result
E2EHandler::HandleCheckStatus( std::uint16_t serviceId,
                               std::uint16_t instanceId,
                               std::uint16_t eventId,
                               E2E_state_machine::E2ECheckStatus checkStatus ) {
    return E2EHandlerImpl::instance().HandleCheckStatus( serviceId, instanceId, eventId,
                                                         checkStatus );
}

bool
E2EHandler::ProtectEvent( const std::uint16_t          serviceId,
                          const std::uint16_t          instanceId,
                          const std::uint16_t          eventId,
                          const crc::Buffer& inputBuffer,
                          crc::Buffer&       outputBuffer ) {
    return E2EHandlerImpl::instance().ProtectEvent( serviceId, instanceId, eventId,
                                                    inputBuffer, outputBuffer );
}

Result
E2EHandler::CheckEvent( const std::uint16_t          serviceId,
                        const std::uint16_t          instanceId,
                        const std::uint16_t          eventId,
                        const crc::Buffer& inputBuffer,
                        crc::Buffer&       outputBuffer ) {
    return E2EHandlerImpl::instance().CheckEvent( serviceId, instanceId, eventId,
                                                  inputBuffer, outputBuffer );
}

bool
E2EHandler::ProtectMethod( const std::uint16_t          serviceId,
                           const std::uint16_t          instanceId,
                           const std::uint16_t          methodId,
                           const crc::Buffer& inputBuffer,
                           crc::Buffer&       outputBuffer ) {
    return E2EHandlerImpl::instance().ProtectMethod( serviceId, instanceId, methodId,
                                                     inputBuffer, outputBuffer );
}

Result
E2EHandler::CheckMethod( const std::uint16_t          serviceId,
                         const std::uint16_t          instanceId,
                         const std::uint16_t          methodId,
                         const crc::Buffer& inputBuffer,
                         crc::Buffer&       outputBuffer ) {
    return E2EHandlerImpl::instance().CheckMethod( serviceId, instanceId, methodId,
                                                   inputBuffer, outputBuffer );
}

bool
E2EHandler::IsProtected( const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventOrMethodId ) {
    return E2EHandlerImpl::instance().IsProtected(serviceId, instanceId, eventOrMethodId);
}

void
E2EHandler::GetDataIdAndCounter( const std::uint16_t          serviceId,
                                 const std::uint16_t          instanceId,
                                 const std::uint16_t          eventId,
                                 const crc::Buffer& inputBuffer,
                                 DataIdentifier&              dataId,
                                 std::uint32_t&               counter ) {
    return E2EHandlerImpl::instance().GetDataIdAndCounter( serviceId, instanceId, eventId, inputBuffer, dataId, counter );
}

}  // namespace e2exf
