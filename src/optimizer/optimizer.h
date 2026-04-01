#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "frame.h"
#include "landmark.h"
#include "map.h"

class Optimizer{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    g2o::SparseOptimizer optimizer;
    Optimizer(bool verbose);
    void optimizePose(std::shared_ptr<Frame> frame);
    void optimizeBundle(std::vector<std::shared_ptr<Frame>>frame);
    void localBundleAdjustment(std::shared_ptr<Frame> frame, std::shared_ptr<Map>currentMap);
};

