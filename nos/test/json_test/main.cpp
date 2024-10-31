#include <iostream>
#include "json/json.h"
#include <iomanip>


 
using namespace std;
 

char *getfileAll(const char *fname)
{
	FILE *fp;
	char *str;
	char txt[1000];
	int filesize;
	if ((fp=fopen(fname,"r"))==NULL){
		printf("open file %s fail \n",fname);
		return NULL;
	}
 
	/*
	获取文件的大小
	ftell函数功能:得到流式文件的当前读写位置,其返回值是当前读写位置偏离文件头部的字节数.
	*/
	fseek(fp,0,SEEK_END); 
	filesize = ftell(fp);
 
	str=(char *)malloc(filesize);
	str[0]=0;
 
	rewind(fp);
	while((fgets(txt,1000,fp))!=NULL){
		strcat(str,txt);
	}
	fclose(fp);
	return str;
}
 

int writefileAll(const char* fname, const char* data)
{
	FILE *fp;
	if ((fp=fopen(fname, "w")) == NULL)
	{
		printf("open file %s fail \n", fname);
		return 1;
	}
	
	fprintf(fp, "%s", data);
	fclose(fp);
	
	return 0;
}
 

int parseJSON(const char* jsonstr)
{
	Json::Reader reader;
	Json::Value  rootValue;
 
	if (!reader.parse(jsonstr, rootValue, false))
	{
		printf("bad json format!\n");
		return 1;
	}
	std::uint32_t Index = rootValue["Index"].asUInt();

    std::cout << "Index = " << Index << std::endl;

 
	Json::Value & resultValue = rootValue["ResultValue"];
	for (uint32_t i=0; i<resultValue.size(); i++)
	{
		Json::Value subJson = resultValue[i];

        std::uint64_t test_uint64Data = subJson["test_uint64Data"].asUInt64();
		std::cout << "test_uint64Data = " << test_uint64Data << std::endl;

        std::int32_t test_intData = subJson["test_intData"].asInt();
		std::cout << "test_intData = " << test_intData << std::endl;

		bool test_boolData = subJson["test_boolData"].asBool();
		std::cout << "test_boolData = " << test_boolData << std::endl;

		const char * test_cStringData = subJson["test_cStringData"].asCString();
		std::cout << "test_cStringData = " << test_cStringData << std::endl;

		std::string test_cppStringData = subJson["test_cppStringData"].asString();
		std::cout << "test_cppStringData = " << test_cppStringData.c_str() << std::endl;

		//Hex data.
		std::uint32_t  test_HexData = std::strtoul(subJson["test_HexData"].asString().c_str(), 0, 16);
		std::cout << "test_HexData in dec = " << test_HexData  << ", in hex = " << std::setbase(16) << test_HexData  << std::setbase(10) << std::endl;

		double test_DobleData = subJson["test_DobleData"].asDouble();
		std::cout << "test_DobleData = " << std::setiosflags(std::ios::fixed) << std::setprecision(6) << test_DobleData << std::endl;

		float test_FloatData = subJson["test_FloatData"].asFloat();
		std::cout << "test_FloatData = " << std::setiosflags(std::ios::fixed) << std::setprecision(6) << test_FloatData << std::endl;

	}
	return 0;
}
 

int createJSON()
{
	Json::Value req;
	req["Result"] = 1;
	req["ResultMessage"] = "200";
 
	Json::Value	object1;
	object1["cpuRatio"] = "4.04";
	object1["serverIp"] = "42.159.116.104";
	object1["conNum"] = "1";
	object1["websocketPort"] = "0";
	object1["mqttPort"] = "8883";
	object1["TS"] = "1504665880572";
	Json::Value	object2;
	object2["cpuRatio"] = "2.04";
	object2["serverIp"] = "42.159.122.251";
	object2["conNum"] = "2";
	object2["websocketPort"] = "0";
	object2["mqttPort"] = "8883";
	object2["TS"] = "1504665896981";
 
	Json::Value jarray;
	jarray.append(object1);
	jarray.append(object2);
 
	req["ResultValue"] = jarray;
 
	Json::FastWriter writer;
	string jsonstr = writer.write(req);
 
	printf("%s\n", jsonstr.c_str());
 
 	std::string json_file = "createJson.json";
	writefileAll(json_file.c_str(), jsonstr.c_str());
	return 0;
}
 
int main()
{
	/*读取Json串，解析Json串*/
	std::string json_path = "./parseJson.json";
	char* json = getfileAll(json_path.c_str());
	parseJSON(json);
	printf("===============================\n");
 
	/*组装Json串*/
	createJSON();
 
	getchar();
	return 1;
}
