#include "atlas.h"


Atlas::Atlas(){};
std::shared_ptr<Map> Atlas:: initiateNewMap(){
    std::shared_ptr<Map> map= std::make_shared<Map>();
    this->map=map;
    maps.push_back(map);
    return map;
}
std::shared_ptr<Map> Atlas::getActiveMap(){
    return map;
}