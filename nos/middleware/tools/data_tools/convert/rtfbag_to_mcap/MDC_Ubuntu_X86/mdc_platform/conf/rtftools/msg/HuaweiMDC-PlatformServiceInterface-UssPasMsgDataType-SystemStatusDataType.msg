{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssPasMsgDataType/SystemStatusDataType",
  "Namespace": "::mdc::passystemstatus",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssComMsgDataType/Header"
    },
    {
      "Name": "version",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssPasMsgDataType/Version"
    },
    {
      "Name": "msgModRollingCounter",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "msgStatusRollingCounter",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "workModStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "frontWorkStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "rearWorkStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "sideWorkStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "displayCmd",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "autoModStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "reserveMsg1",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "reserveMsg2",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    }
  ],
  "IsPlane": "false"
}