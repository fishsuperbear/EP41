#include "mcap_common.hpp"

namespace hozon {
namespace netaos {
namespace mcap {

std::string GetDirName(std::string path) {
    size_t pos = path.rfind('/');
	if (pos != std::string::npos) {
		return path.substr(0, pos);
	} else {
		return "";
	}
}

std::string GetFileName(std::string path) {
	size_t length = path.length();
	size_t pos = path.rfind('/');
	if (pos != std::string::npos) {
		return path.substr(pos + 1, length - pos -1);
	} else {
		return path;
	}
}

}  // namespace mcap
}  // namespace netaos
}  // namespace hozon
