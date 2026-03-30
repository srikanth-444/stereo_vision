#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "triangulation.h"

namespace py = pybind11;

PYBIND11_MODULE(Stereo, m) {
    py::class_<Stereo>(m, "Stereo")
        .def(py::init<Eigen::Matrix3f&,Eigen::Matrix4f&,Eigen::Vector4f,
                Eigen::Matrix3f,Eigen::Matrix4f,Eigen::Vector4f,int ,int>())
        .def("getDepth",
            [](Stereo &self,
               std::shared_ptr<Frame> leftFrame,
               std::shared_ptr<Frame> rightFrame,
               float epipolar_threshold)
            {
                std::vector<Eigen::Vector3f> points_3d;
                std::vector<int> featureIds;

                self.getDepth(leftFrame, rightFrame, epipolar_threshold,
                              points_3d, featureIds);

                return py::make_tuple(points_3d, featureIds);
            }
        )
        .def("rectifyPoints",&Stereo::rectifyPoints)
        .def("rectifyImage",&Stereo::rectifyImage);
}