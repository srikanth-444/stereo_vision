#include <pybind11/pybind11.h>
#include "featureExtractor.h"

namespace py = pybind11;

// forward declarations
void bind_orb(py::module_ &m);


PYBIND11_MODULE(FeatureExtractor, m) {
    py::class_<FeatureExtractor,std::shared_ptr<FeatureExtractor>>(m, "FeatureExtractor");
    bind_orb(m);
}