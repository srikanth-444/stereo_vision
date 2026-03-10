#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "orb.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array mat_to_numpy_copy(const cv::Mat& mat)
{
    if (!mat.data)
        throw std::runtime_error("Empty matrix");

    // cv::Mat mat = mat.clone(); // make sure we own the memory
    size_t elem_size = mat.elemSize1();

    if (mat.channels() == 1)
    {
        return py::array(
            { mat.rows, mat.cols },
            { mat.step[0], elem_size },
            mat.data
        );
    }
    else if (mat.channels() == 3)
    {
        return py::array(
            { mat.rows, mat.cols, 3 },
            { mat.step[0], elem_size * 3, elem_size },
            mat.data
        );
    }

    throw std::runtime_error("Unsupported channel count");
}


py::array_t<float> keypoints_to_numpy(const std::vector<cv::KeyPoint> &kps) {
    // 1. Explicitly define the shape
    std::vector<size_t> shape = { kps.size(), 6 };
    
    // 2. Use the typed array_t constructor
    py::array_t<float> result(shape);
    
    // 3. Use a proxy for high-speed, direct memory access
    auto buf = result.mutable_unchecked(); 

    for (size_t i = 0; i < kps.size(); ++i) {
        buf(i, 0) = kps[i].pt.x;
        buf(i, 1) = kps[i].pt.y;
        buf(i, 2) = kps[i].size;
        buf(i, 3) = kps[i].angle;
        buf(i, 4) = kps[i].response;
        buf(i, 5) = static_cast<float>(kps[i].octave);
    }

    return result;
}


PYBIND11_MODULE(orb_slam_features, m) {
    

    py::class_<ORB_SLAM3::ORBextractor>(m, "ORBExtractor")
        .def(py::init<int, float, int, int, int>(),
             py::arg("nfeatures") = 1000,
             py::arg("scaleFactor") = 1.2,
             py::arg("nlevels") = 8,
             py::arg("iniThFAST") = 20,
             py::arg("minThFAST") = 7)
        .def("__call__", [](ORB_SLAM3::ORBextractor &self, py::array input_img) {

            py::buffer_info buf = input_img.request();
            // std::cout << "[DEBUG] Num dimensions: " << buf.ndim << std::endl;
            // std::cout << "[DEBUG] Shape: ";
            // for (int i = 0; i < buf.ndim; ++i) std::cout << buf.shape[i] << " ";
            // std::cout << std::endl;
            // std::cout << "[DEBUG] Format: " << buf.format << std::endl;

            if (buf.ndim != 2 && buf.ndim != 3)
                throw std::runtime_error("Image must be 2D or 3D");
            if (buf.format != py::format_descriptor<unsigned char>::format())
                throw std::runtime_error("Image must be uint8");

            int height = buf.shape[0];
            int width  = buf.shape[1];

            cv::Mat img;
            if (buf.ndim == 2)
                img = cv::Mat(height, width, CV_8UC1, buf.ptr);
            else
                img = cv::Mat(height, width, CV_8UC3, buf.ptr);

            // img = img.clone();  // safe copy
            // std::cout << "[DEBUG] CV type: " << img.type() << ", rows=" << img.rows << ", cols=" << img.cols << std::endl;
            
            
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            std::vector<int> lap_area = {0, img.cols};
            
            {
            py::gil_scoped_release release;
            self(img, cv::noArray(), keypoints, descriptors, lap_area);
            // std::cout << "[DEBUG] Number of keypoints: " << keypoints.size() << std::endl;
            // std::cout << "[DEBUG] Descriptor size: " << descriptors.rows << " x " << descriptors.cols << std::endl;
            }
            py::array desc_np = mat_to_numpy_copy(descriptors);

            py::array keypoints_np = keypoints_to_numpy(keypoints);

            return py::make_tuple(keypoints_np, desc_np);
        });

    // We also need to bind cv::KeyPoint so Python can read the list
    // py::class_<cv::KeyPoint>(m, "KeyPoint")
    //     .def_readwrite("pt", &cv::KeyPoint::pt)
    //     .def_readwrite("angle", &cv::KeyPoint::angle)
    //     .def_readwrite("response", &cv::KeyPoint::response)
    //     .def_readwrite("octave", &cv::KeyPoint::octave);
}