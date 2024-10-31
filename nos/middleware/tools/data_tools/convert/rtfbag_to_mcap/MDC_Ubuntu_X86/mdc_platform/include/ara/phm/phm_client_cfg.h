/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Phm client config header
 * Create: 2021-01-28
 */
#ifndef ARA_PHM_CLIENT_CFG_H
#define ARA_PHM_CLIENT_CFG_H

#include <string>

namespace ara {
namespace phm {
/**
 * @defgroup Config Config
 * @brief Container for PhmClientCfg objects.
 * @ingroup Config
 */
/* AXIVION Next Line AutosarC++19_03-A0.1.6 : The class is external interface, it is offered to the user. */
class PhmClientCfg final {
public:
    /**
     * @ingroup Config
     * @brief Set the user of the phm client.
     * @param[in] user The user of the phm client.
     * @return void
     * @req{AR-iAOS-RTF-RTFPHM-00047,
     * RTFPHM allows the client to report faults to a specified PHM instance.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00047
     * }
     */
    static void SetUser(std::string const &user);
    /**
     * @ingroup Config
     * @brief Get the user of the phm client.
     * @return std::string
     * @req{AR-iAOS-RTF-RTFPHM-00047,
     * RTFPHM allows the client to report faults to a specified PHM instance.,
     * D,
     * DR-iAOS-RTF-RTFPHM-00047
     * }
     */
    static std::string GetUser();
};
} // namespace phm
} // namespace ara
#endif

