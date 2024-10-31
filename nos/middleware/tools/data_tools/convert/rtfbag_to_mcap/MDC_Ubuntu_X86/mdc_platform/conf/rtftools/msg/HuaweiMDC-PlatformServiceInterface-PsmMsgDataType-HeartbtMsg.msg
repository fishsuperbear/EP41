{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/PsmMsgDataType/HeartbtMsg",
  "Namespace": "::mdc::psm",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "type",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "seq",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "procStartTime",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "packageLen",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "version",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "devId",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "license",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "state",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "hbState",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "psmMode",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "psmCfgRole",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "psmRtRole",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "switchDisable",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "switchState",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "reserve1",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "faultType",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint32_t"
    },
    {
      "Name": "faultParaNum",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint16_t"
    },
    {
      "Name": "faultPara",
      "Type": "/HuaweiMDC/PlatformServiceInterface/PsmMsgDataType/FaultParaArray"
    },
    {
      "Name": "sendTime",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint64_t"
    },
    {
      "Name": "reserve2",
      "Type": "/HuaweiMDC/StdCppBasicDataType/uint64_t"
    }
  ],
  "IsPlane": "false"
}