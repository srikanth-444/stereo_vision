#include "map.h"

class Atlas{
    public:
    std::vector<std::shared_ptr<Map>> maps;
    std::shared_ptr<Map> map;
    Atlas();
    std::shared_ptr<Map> initiateNewMap();
    std::shared_ptr<Map> getActiveMap();
};