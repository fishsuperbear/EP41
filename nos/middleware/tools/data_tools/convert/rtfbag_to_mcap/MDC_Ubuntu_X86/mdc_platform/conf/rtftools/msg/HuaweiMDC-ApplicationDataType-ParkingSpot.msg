{
  "ShortName": "/HuaweiMDC/ApplicationDataType/ParkingSpot",
  "Namespace": "",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "parking_class",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "class_confidence",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "parking_spot_ego",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "parking_spot_enu",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "entry",
      "Type": "/HuaweiMDC/ApplicationDataType/Uint8Array"
    },
    {
      "Name": "parking_spot_type",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    }
  ],
  "IsPlane": "false"
}