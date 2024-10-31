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
#include <cstring>
#include <string>
#include <iostream>
#include <type_traits>
#include "ne_someip_log.h"
#include "cJSON.h"
#include "extend/e2e/profiles.h"
#include "config_helpers.h"
#include "config_reader.h"

namespace {

using ProfileName    = profile::ProfileName;
using DataIdMode     = profile::profile11::DataIdMode;
using DataIdentifier = e2exf::DataIdentifier;

struct ProfileNameTranslator {
    using internal_type = std::string;
    using external_type = ProfileName;

    static ProfileName get_value( const std::string& name ) {
        static const std::map<std::string, ProfileName> names{
            {"PROFILE_01", ProfileName::PROFILE_01}, {"PROFILE_02", ProfileName::PROFILE_02},
            {"PROFILE_04", ProfileName::PROFILE_04}, {"PROFILE_05", ProfileName::PROFILE_05},
            {"PROFILE_06", ProfileName::PROFILE_06}, {"PROFILE_07", ProfileName::PROFILE_07},
            {"PROFILE_11", ProfileName::PROFILE_11}, {"PROFILE_22", ProfileName::PROFILE_22}};

        auto it = names.find( name );
        if ( it == names.end() ) {
            throw std::out_of_range( "profile is no supported" );
            return ProfileName::PROFILE_UNKNOW;
        }
        return it->second;
    }
};

struct DataIdModeTranslator {
    using internal_type = std::string;
    using external_type = DataIdMode;

    static DataIdMode get_value( const std::string& mode ) {
        if ( mode == "0" ) {
            return DataIdMode::ALL_16_BIT;
        }
        if ( mode == "1" ) {
            return DataIdMode::LOWER_12_BIT;
        }
        throw std::out_of_range( "invalid data ID mode" );
    }
};

template <typename T>
struct SafeUnsignedTranslator {
    using internal_type = std::string;
    using external_type = T;

    static_assert( std::is_unsigned<T>::value, "type is not unsigned" );

    static T get_value( const std::string& param ) {
        const auto         value    = std::strtoull( param.c_str(), nullptr, 10 );
        static constexpr T maxValue = std::numeric_limits<T>::max();
        if ( value > maxValue ) {
            ne_someip_log_error("value:[%u], maxValue:[%u]", value, maxValue);
            throw std::invalid_argument{"Parameter value exceeds the max value of type."};
        }
        return static_cast<T>( value );
    }
};

void LoadE2EXfConfiguration( cJSON*  jsonObject,
                             e2exf::Config& config,
                             std::string&       propSetName ) {
    using e2exf::RegisterProfile;
    using e2exf::RegisterStateMachine;
    using namespace profile;

    if (cJSON_False == cJSON_IsInvalid(jsonObject) ||
        cJSON_NULL ==  cJSON_GetObjectItem(jsonObject, propSetName.c_str())->type) {
        throw std::runtime_error( "jsonObject is unValid." );
    }

    // End2EndEventProtectionProps
    std::string key_dataId( "dataId" );
    std::string key_maxDataLength( "maxDataLength" );
    std::string key_minDataLength( "minDataLength" );
    // E2EProfileConfiguration
    std::string key_dataIdMode( "dataIdMode" );
    std::string key_maxDeltaCounter( "maxDeltaCounter" );
    std::string key_maxErrorStateInit( "maxErrorStateInit" );
    std::string key_maxErrorStateInvalid( "maxErrorStateInvalid" );
    std::string key_maxErrorStateValid( "maxErrorStateValid" );
    std::string key_minOkStateInit( "minOkStateInit" );
    std::string key_minOkStateInvalid( "minOkStateInvalid" );
    std::string key_minOkStateValid( "minOkStateValid" );
    std::string key_profileName( "profileName" );
    std::string key_windowSize( "windowSize" );
    // parameters below are not descirbed in manifest but needed according PRS_E2EProtocol
    // required by profiles: 4,5,6,7,22
    std::string key_offset( "offset" );
    // required by profile 22
    std::string key_dataIdList( "DataIdList" );

    ne_someip_log_debug("StateMachinesConfiguration is valid type[%u], size[%u]",
        cJSON_GetObjectItem(jsonObject, propSetName.c_str())->type, cJSON_GetArraySize(cJSON_GetObjectItem(jsonObject, propSetName.c_str())));

    if ( cJSON_NULL == cJSON_GetObjectItem(jsonObject, propSetName.c_str())->type) {
        throw std::runtime_error( "failed to find root element in E2E configuration" );
    }

    for ( int32_t i = 0U; i < cJSON_GetArraySize(cJSON_GetObjectItem(jsonObject, propSetName.c_str())); i++ ) {

        const auto dataIdentifier = SafeUnsignedTranslator<std::uint32_t>::get_value(
            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_dataId.c_str()))
            );

        // both variants are enabled
        const bool isProtector = true;
        const bool isChecker   = true;

        try {
            const auto profileNum = ProfileNameTranslator::get_value(
                cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_profileName.c_str()))
                );
            switch ( profileNum ) {
                case ProfileName::PROFILE_04: {
                    const profile04::Config profileConfig(
                        dataIdentifier,
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_offset.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minDataLength.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDataLength.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDeltaCounter.c_str()))
                            ) );

                    RegisterProfile<profile04::Protector, profile04::Checker>(
                        config, dataIdentifier, profileConfig, isProtector, isChecker );
                } break;
                case ProfileName::PROFILE_05: {
                    const profile05::Config profileConfig(
                        dataIdentifier, profile05::Profile05::dataLength,
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDeltaCounter.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_offset.c_str()))
                            ) );

                    RegisterProfile<profile05::Protector, profile05::Checker>(
                        config, dataIdentifier, profileConfig, isProtector, isChecker );
                } break;
                case ProfileName::PROFILE_06: {
                    const profile06::Config profileConfig{
                        static_cast<uint16_t>(dataIdentifier),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minDataLength.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDataLength.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDeltaCounter.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_offset.c_str()))
                            )};

                    RegisterProfile<profile06::Protector, profile06::Checker>(
                        config, dataIdentifier, profileConfig, isProtector, isChecker );
                } break;
                case ProfileName::PROFILE_07: {
                    const profile07::Config profileConfig{
                        dataIdentifier,
                        SafeUnsignedTranslator<std::uint32_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDataLength.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minDataLength.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDeltaCounter.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_offset.c_str()))
                            )};

                    RegisterProfile<profile07::Protector, profile07::Checker>(
                        config, dataIdentifier, profileConfig, isProtector, isChecker );
                } break;
                case ProfileName::PROFILE_11: {
                    bool dataIdModeExistFlag =
                        cJSON_HasObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_dataIdMode.c_str());
                    if ( !dataIdModeExistFlag ) {
                        throw std::invalid_argument{
                            std::string(
                                "Configuration parameter dataIdMode is not correctly set" )
                                .c_str()};
                    }

                    const profile11::Config profileConfig(
                        profile11::Profile11::counterOffset, profile11::Profile11::crcOffset,
                        dataIdentifier,
                        DataIdModeTranslator::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_dataIdMode.c_str()))
                            ),
                        profile11::Profile11::dataIdNibbleOffset, profile11::Profile11::dataLength,
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDeltaCounter.c_str()))
                            ) );

                    RegisterProfile<profile11::Protector, profile11::Checker>(
                        config, dataIdentifier, profileConfig, isProtector, isChecker );
                } break;
                case ProfileName::PROFILE_22: {
                    // required by profile 22
                    std::array<std::uint8_t, 16> dataIdList;
                    bool                               dataIdModeExistFlag =
                        cJSON_HasObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_dataIdList.c_str());
                    if ( !dataIdModeExistFlag ) {
                        throw std::invalid_argument{
                            std::string(
                                "Configuration parameter DataIdList is not correctly set" )
                                .c_str()};
                    } else {
                        uint8_t idx{0U};
                        for ( int32_t j = 0U;
                              j < cJSON_GetArraySize(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_dataIdList.c_str()));
                              j++ ) {
                            dataIdList[ idx ] = static_cast<std::uint8_t>(
                                cJSON_GetNumberValue(cJSON_GetArrayItem(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_dataIdList.c_str()), j))
                                );
                            ++idx;
                        }
                    }

                    const profile22::Config profileConfig(
                        dataIdentifier, dataIdList, profile22::Profile22::dataLength,
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxDeltaCounter.c_str()))
                            ),
                        SafeUnsignedTranslator<std::uint16_t>::get_value(
                            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_offset.c_str()))
                            ) );

                    RegisterProfile<profile22::Protector, profile22::Checker>(
                        config, dataIdentifier, profileConfig, isProtector, isChecker );
                } break;
                default:
                    throw std::invalid_argument( "unsupported profile" );
            }  // switch

            const E2E_state_machine::Config stateMachineConfig{
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_windowSize.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minOkStateInit.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxErrorStateInit.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minOkStateValid.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxErrorStateValid.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minOkStateInvalid.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxErrorStateInvalid.c_str()))
                    )};
            RegisterStateMachine( config, dataIdentifier, stateMachineConfig );

        } catch ( const std::invalid_argument& e ) {
            ne_someip_log_debug("Could not create config for %u Id. %s",
                static_cast<std::uint32_t>( dataIdentifier ), e.what());
            throw;
        }
    }
}

e2exf::Config GetE2EXfConfiguration(cJSON* jsonObject ) {
    using e2exf::RegisterProfile;
    using e2exf::RegisterStateMachine;
    using namespace profile;

    std::string eventPropsSet( "End2EndEventProtectionPropsSet" );
    std::string methodPropsSet( "End2EndMethodProtectionPropsSet" );
    e2exf::Config config;

    if (cJSON_False == cJSON_IsInvalid(jsonObject)) {
        throw std::runtime_error( "jsonObject is unValid." );
    } else {
        // add event state machine
        LoadE2EXfConfiguration( jsonObject, config, eventPropsSet );

        // add method state machine
        LoadE2EXfConfiguration( jsonObject, config, methodPropsSet );
    }

    return config;
}


using dataid_mapping_type_t =
    std::map<std::tuple<std::uint16_t, std::uint16_t, std::uint16_t>, DataIdentifier>;

void LoadDataIdMapping( cJSON* jsonObject,
                        dataid_mapping_type_t&  result,
                        std::string&      dataSetName,
                        bool                    eventOrMethod ) {
    if (cJSON_False == cJSON_IsInvalid(jsonObject) ||
        cJSON_NULL == cJSON_GetObjectItem(jsonObject, dataSetName.c_str())->type) {
        throw std::runtime_error( "jsonObject is unValid." );
    }

    std::string key_serviceId( "serviceId" );
    std::string key_instanceId( "instanceId" );
    std::string key_eventId( "eventId" );
    std::string key_methodId( "methodId" );
    std::string key_dataId( "dataId" );

    ne_someip_log_debug("DataIdConfiguration is valid type[%u], size[%u]",
        cJSON_GetObjectItem(jsonObject, dataSetName.c_str())->type, cJSON_GetArraySize(cJSON_GetObjectItem(jsonObject, dataSetName.c_str())));

    if (cJSON_NULL == cJSON_GetObjectItem(jsonObject, dataSetName.c_str())->type) {
        throw std::runtime_error( "failed to find root element in dataid configuration" );
    }

    for ( int32_t i = 0U; i < cJSON_GetArraySize(cJSON_GetObjectItem(jsonObject, dataSetName.c_str())); i++ ) {
        const auto dataId = SafeUnsignedTranslator<std::uint32_t>::get_value(
            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, dataSetName.c_str()), i), key_dataId.c_str()))
            );
        const auto serviceId = SafeUnsignedTranslator<std::uint16_t>::get_value(
            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, dataSetName.c_str()), i), key_serviceId.c_str()))
            );
        const auto instanceId = SafeUnsignedTranslator<std::uint16_t>::get_value(
            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, dataSetName.c_str()), i), key_instanceId.c_str()))
            );
        std::uint16_t eventOrMethodId;
        if ( eventOrMethod ) {
            eventOrMethodId = SafeUnsignedTranslator<std::uint16_t>::get_value(
                cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, dataSetName.c_str()), i), key_eventId.c_str()))
                );
        } else {
            eventOrMethodId = SafeUnsignedTranslator<std::uint16_t>::get_value(
                cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, dataSetName.c_str()), i), key_methodId.c_str()))
                );
        }

        auto key = std::make_tuple( serviceId, instanceId, eventOrMethodId );
        if ( 0 != result.count( key ) ) {
            throw std::invalid_argument(
                "Wrong configuration of dataid mapping: overlapping configuration" );
        }

        result[ key ] = dataId;
    }
}

dataid_mapping_type_t GetDataIdMapping(cJSON* jsonObject ) {
    std::string eventDataSet( "End2EndEventDataIdMappingSet" );
    std::string methodDataSet( "End2EndMethodDataIdMappingSet" );
    dataid_mapping_type_t result;

    if (NULL == jsonObject || cJSON_False == cJSON_IsInvalid(jsonObject)) {
        throw std::runtime_error( "jsonObject is unValid." );
    } else {
        // add event dataid
        LoadDataIdMapping( jsonObject, result, eventDataSet, true );

        // add method dataid
        LoadDataIdMapping( jsonObject, result, methodDataSet, false );
    }

    return result;
}

void LoadStateMachinesConfiguration(cJSON*  jsonObject,
                                     e2exf::Config& config,
                                     std::string&       propSetName ) {
    if ( cJSON_False == cJSON_IsInvalid(jsonObject) ||
        cJSON_NULL == cJSON_GetObjectItem(jsonObject, propSetName.c_str())->type) {
        throw std::runtime_error( "jsonObject is unValid." );
    }
    std::string key_dataId( "dataId" );
    std::string key_maxDataLength( "maxDataLength" );
    std::string key_minDataLength( "minDataLength" );
    std::string key_maxDeltaCounter( "maxDeltaCounter" );
    std::string key_maxErrorStateInit( "maxErrorStateInit" );
    std::string key_maxErrorStateInvalid( "maxErrorStateInvalid" );
    std::string key_maxErrorStateValid( "maxErrorStateValid" );
    std::string key_minOkStateInit( "minOkStateInit" );
    std::string key_minOkStateInvalid( "minOkStateInvalid" );
    std::string key_minOkStateValid( "minOkStateValid" );
    std::string key_profileName( "profileName" );
    std::string key_windowSize( "windowSize" );
    std::string key_offset( "offset" );

    ne_someip_log_debug("StateMachinesConfiguration is valid type[%u], size[%u]",
        cJSON_GetObjectItem(jsonObject, propSetName.c_str())->type, cJSON_GetArraySize(cJSON_GetObjectItem(jsonObject, propSetName.c_str())));

    if (cJSON_NULL == cJSON_GetObjectItem(jsonObject, propSetName.c_str())->type) {
        throw std::runtime_error( "failed to find root element in E2E configuration" );
    }

    for ( int32_t i = 0U; i < cJSON_GetArraySize(cJSON_GetObjectItem(jsonObject, propSetName.c_str())); i++ ) {
        const auto dataIdentifier = SafeUnsignedTranslator<std::uint32_t>::get_value(
            cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_dataId.c_str()))
            );
        try {
            const E2E_state_machine::Config stateMachineConfig{
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_windowSize.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minOkStateInit.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxErrorStateInit.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minOkStateValid.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxErrorStateValid.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_minOkStateInvalid.c_str()))
                    ),
                SafeUnsignedTranslator<std::uint8_t>::get_value(
                    cJSON_GetStringValue(cJSON_GetObjectItem(cJSON_GetArrayItem(cJSON_GetObjectItem(jsonObject, propSetName.c_str()), i), key_maxErrorStateInvalid.c_str()))
                    )};
            RegisterStateMachine( config, dataIdentifier, stateMachineConfig );
        } catch ( const std::invalid_argument& e ) {
            ne_someip_log_debug("Could not create config for %u Id. %s",
                static_cast<std::uint32_t>( dataIdentifier ), e.what());
            throw;
        }
    }
}

using state_machines_t = std::map<e2exf::DataIdentifier,
                                        std::shared_ptr<E2E_state_machine::StateMachine>>;
state_machines_t GetStateMachinesConfiguration(cJSON* jsonObject ) {
    using e2exf::RegisterProfile;
    using e2exf::RegisterStateMachine;
    using namespace profile;
    e2exf::Config config;

    std::string eventPropsSet( "End2EndEventProtectionPropsSet" );
    std::string methodPropsSet( "End2EndMethodProtectionPropsSet" );

    if (cJSON_False == cJSON_IsInvalid(jsonObject)) {
        throw std::runtime_error( "jsonObject is unValid." );
    } else {
        // add event state machine
        LoadStateMachinesConfiguration( jsonObject, config, eventPropsSet );

        // add method state machine
        LoadStateMachinesConfiguration( jsonObject, config, methodPropsSet );
    }

    return config.stateMachines;
}

template <typename ResultType, typename ProcessorType>
ResultType LoadConfiguration( const std::string&               fileName,
                              e2exf::ConfigurationFormat   configurationFormat,
                              ProcessorType                          processor,
                              e2exf::ConfigurationFileType configurationFileType ) {
    if ( fileName.empty() ) {
        throw std::invalid_argument( "Empty filename given, cannot load configuration." );
    }

    if (e2exf::ConfigurationFormat::JSON != configurationFormat) {
        throw std::invalid_argument( "configuration Format is not json, cannot load configuration." );
    }

    switch ( configurationFileType ) {
        case e2exf::ConfigurationFileType::E2E_FILE_TYPE_SM:
            break;
        case e2exf::ConfigurationFileType::E2E_FILE_TYPE_DATAID:
            break;
        case e2exf::ConfigurationFileType::E2E_FILE_TYPE_E2EXF:
            break;
        default:
            throw std::out_of_range{"Unknown configuration format."};
    }

    cJSON* jsonObject;
    FILE* file_fd = fopen(fileName.data(), "rb");
    if (NULL == file_fd) {
        throw std::invalid_argument( "configuration file open failed." );
    }
    fseek(file_fd, 0, SEEK_END);
    long data_length = ftell(file_fd);
    rewind(file_fd);

    char* file_data = (char*)malloc(data_length + 1);
    memset(file_data, 0, data_length + 1);

    fread(file_data, 1, data_length, file_fd);
    fclose(file_fd);

    jsonObject = cJSON_Parse(file_data);
    if (NULL != file_data) {
        free(file_data);
        file_data = NULL;
    }

    try {
        return processor( jsonObject );
    } catch(const std::exception& e) {
        ne_someip_log_debug("LoadConfiguration error: %s", e.what());
        throw;
    } catch(...) {
        ne_someip_log_debug("LoadConfiguration error");
        throw;
    }
}

}  // anonymous namespace

namespace e2exf {

Config LoadE2EConfiguration( const std::string& fileName, ConfigurationFormat format, bool& ret ) {
    e2exf::Config config;
    try {
        ret = true;
        return LoadConfiguration<Config>( fileName, format, GetE2EXfConfiguration,
                                      e2exf::ConfigurationFileType::E2E_FILE_TYPE_E2EXF );
    } catch(const std::exception& e) {
        ne_someip_log_debug("LoadE2EConfiguration error: %s", e.what());
        ret = false;
        return config;
    } catch(...)  {
        ne_someip_log_debug("LoadE2EConfiguration error");
        ret = false;
        return config;
    }
}

dataid_mapping_type_t LoadE2EDataIdMapping( const std::string& fileName,
                                            ConfigurationFormat      format,
                                            bool& ret ) {
    dataid_mapping_type_t value;
    try {
        ret = true;
        return LoadConfiguration<dataid_mapping_type_t>(
            fileName, format, GetDataIdMapping,
            e2exf::ConfigurationFileType::E2E_FILE_TYPE_DATAID );
    } catch(const std::exception& e) {
        ne_someip_log_debug("LoadE2EDataIdMapping error: %s", e.what());
        ret = false;
    } catch(...)  {
        ne_someip_log_debug("LoadE2EDataIdMapping error");
        ret = false;
    }
    return value;
}

state_machines_t LoadE2EStateMachines( const std::string& fileName,
                                       ConfigurationFormat      format,
                                       bool& ret ) {
    state_machines_t value;
    try {
        ret = true;
        return LoadConfiguration<state_machines_t>(
            fileName, format, GetStateMachinesConfiguration,
            e2exf::ConfigurationFileType::E2E_FILE_TYPE_SM );
    } catch(const std::exception& e) {
        ne_someip_log_debug("LoadE2EStateMachines error: %s", e.what());
        ret = false;
    } catch(...)  {
        ne_someip_log_debug("LoadE2EStateMachines error");
        ret = false;
    }
    return value;
}

}  // namespace e2exf
