{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/LaneLineArrayMsgDataType/Lane",
  "Namespace": "::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "cls",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "clsDescription",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "clsConfidence",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "laneFit",
      "Type": "/HuaweiMDC/FunctionalSoftware/LaneLineArrayMsgDataType/LaneFitParam"
    },
    {
      "Name": "startPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/Pointxy"
    },
    {
      "Name": "endPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/Pointxy"
    },
    {
      "Name": "keyPoints",
      "Type": "/HuaweiMDC/ApplicationDataType/PointxyArray"
    },
    {
      "Name": "homographyMat",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "homographyMatInv",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "timeCreation",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonTime"
    }
  ],
  "IsPlane": "false"
}