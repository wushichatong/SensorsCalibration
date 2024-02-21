#include <ceres/ceres.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>

// 定义刚体变换残差结构体
struct RigidTransformResidual {
    RigidTransformResidual(const Eigen::Vector3d& model_point, const Eigen::Vector3d& measured_point)
        : model_point_(model_point), measured_point_(measured_point) {}

    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, T* residual) const {
        // 刚体变换：旋转 + 平移
        Eigen::Matrix<T, 3, 1> transformed_point;
        for (int i = 0; i < 3; ++i) {
            transformed_point(i) = rotation[i] * model_point_(i) + translation[i];
        }

        // 计算残差（测量点和变换后的模型点之差）
        for (int i = 0; i < 3; ++i) {
            residual[i] = transformed_point(i) - measured_point_.cast<T>()(i);
        }

        return true;
    }

private:
    const Eigen::Vector3d model_point_;
    const Eigen::Vector3d measured_point_;
};

int main() {
    // 创建优化问题
    ceres::Problem problem;

    // 添加参数块（旋转和平移）
    double rotation[3] = {0.1, 0.2, 0.3};  // 初始旋转
    double translation[3] = {1.0, 2.0, 3.0};  // 初始平移
    problem.AddParameterBlock(rotation, 3);
    problem.AddParameterBlock(translation, 3);

    // 添加多个点对的残差项（刚体变换）
    std::vector<Eigen::Vector3d> model_points{
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 1.0)
    };

    std::vector<Eigen::Vector3d> measured_points{
        Eigen::Vector3d(2.0, 2.0, 3.0),
        Eigen::Vector3d(3.0, 4.0, 5.0),
        Eigen::Vector3d(4.0, 5.0, 6.0)
    };

    for (size_t i = 0; i < model_points.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<RigidTransformResidual, 3, 3, 3>(
                new RigidTransformResidual(model_points[i], measured_points[i])
            );

        problem.AddResidualBlock(cost_function, nullptr, rotation, translation);
    }

    // 配置优化选项
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    // 运行优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化结果
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "Optimized Rotation: " << rotation[0] << ", " << rotation[1] << ", " << rotation[2] << std::endl;
    std::cout << "Optimized Translation: " << translation[0] << ", " << translation[1] << ", " << translation[2] << std::endl;

    return 0;
}
