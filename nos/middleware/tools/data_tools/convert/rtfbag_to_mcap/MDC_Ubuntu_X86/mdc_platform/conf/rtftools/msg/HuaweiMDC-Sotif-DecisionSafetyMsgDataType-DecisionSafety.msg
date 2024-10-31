{
  "ShortName": "/HuaweiMDC/Sotif/DecisionSafetyMsgDataType/DecisionSafety",
  "Namespace": "::sotif",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "eventID",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "eventLevel",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "type",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "actionName",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "para",
      "Type": "/HuaweiMDC/ApplicationDataType/FloatArray"
    }
  ],
  "IsPlane": "false"
}