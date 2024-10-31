{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/FusionObjectArrayMsgDataType/FusionObject",
  "Namespace": "::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "existProb",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "timeCreation",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonTime"
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
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "absVelocity",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "pointsInObject",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "cipvFlag",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "cameraStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "coordinate",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "rect3d",
      "Type": "/HuaweiMDC/ApplicationDataType/Rect3d"
    },
    {
      "Name": "absoluteRect3d",
      "Type": "/HuaweiMDC/ApplicationDataType/Rect3d"
    },
    {
      "Name": "velocityCovariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "absVelocityCovariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "absAcceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "accelerationCovariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "absAccelerationCovariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "pathPoints",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "lastUpdateTime",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonTime"
    },
    {
      "Name": "blinkerFlag",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "fusionType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "anchorPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "absAnchorPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    }
  ],
  "IsPlane": "false"
}