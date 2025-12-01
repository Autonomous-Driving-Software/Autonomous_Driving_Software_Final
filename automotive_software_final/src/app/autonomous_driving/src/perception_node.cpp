/*
 * perception_node.cpp
 */
#include "autonomous_driving_config.hpp"
#include "perception_node.hpp"
#include <random>
#include <set>
#include <algorithm>
#include <limits>
#include <numeric>

using namespace Eigen;
using namespace std;

PerceptionNode::PerceptionNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options) {

    //QoS init 
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    //===============parameters===============
    //declare parameters(파라미터 등록+초기값 설정)
    this->declare_parameter("autonomous_driving/ns", "");
    this->declare_parameter("autonomous_driving/loop_rate_hz", 100.0);
    this->declare_parameter("autonomous_driving/use_manual_inputs", false);

    // RANSAC Parameters
    this->declare_parameter("autonomous_driving/ransac_max_iterations", 50);
    this->declare_parameter("autonomous_driving/ransac_inlier_threshold", 0.15);
    this->declare_parameter("autonomous_driving/ransac_min_inlier_ratio", 0.9);

    // ROI Parameters
    this->declare_parameter("autonomous_driving/roi_front", 20.0);
    this->declare_parameter("autonomous_driving/roi_rear", 10.0);
    this->declare_parameter("autonomous_driving/roi_left", 3.0);
    this->declare_parameter("autonomous_driving/roi_right", 3.0);

    ProcessParams();

    RCLCPP_INFO(this->get_logger(), "vehicle_namespace: %s", cfg_.vehicle_namespace.c_str());
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", cfg_.loop_rate_hz);
    RCLCPP_INFO(this->get_logger(), "use_manual_inputs: %d", cfg_.use_manual_inputs);

    // get parameters(파라미터 값 가져오기)

    // ROI
    //this->get_parameter("autonomous_driving/roi_front_length", cfg_.param_m_roi_front_length);
    //this->get_parameter("autonomous_driving/roi_rear_length", cfg_.param_m_roi_rear_length);
    //this->get_parameter("autonomous_driving/roi_left_width", cfg_.param_m_roi_left_width);
    //this->get_parameter("autonomous_driving/roi_right_width", cfg_.param_m_roi_right_width);

    // Reference Path
    //this->get_parameter("autonomous_driving/ref_csv_path", cfg_.param_ref_csv_path);

    //===========subscriber init===============
    //(1)s_manual_input_
    s_manual_input_ = 
    this->create_subscription<ad_msgs::msg::VehicleCommand>(
        "/manual_input", qos_profile, std::bind(&PerceptionNode::CallbackManualInput, this, std::placeholders::_1));

    //(2) s_vehicle_state_
    s_vehicle_state_ = 
    this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&PerceptionNode::CallbackVehicleState, this, std::placeholders::_1));

    //(3) s_lane_points_
    s_lane_points_ = 
    this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&PerceptionNode::CallbackLanePoints, this, std::placeholders::_1));

    //(4) s_mission_
    //s_mission_ = 
    //this->create_subscription<ad_msgs::msg::Mission>(
    //    "mission", qos_profile, std::bind(&PerceptionNode::CallbackMission, this, std::placeholders::_1));

    //===========publisher init===============
    //p_vehicle_command_ = 
    //this->create_publisher<ad_msgs::msg::VehicleCommand>(
    //    "vehicle_command", qos_profile);
    p_driving_way_= 
    this->create_publisher<ad_msgs::msg::PolyfitLaneData>(
        "driving_way", qos_profile);

    p_poly_lanes_ = 
    this->create_publisher<ad_msgs::msg::PolyfitLaneDataArray>(
        "poly_lanes", qos_profile);

    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); });
}

PerceptionNode::~PerceptionNode() {}

void PerceptionNode::ProcessParams() {
    //get parameters : 선언한 파라미터 값을 읽어오기
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("autonomous_driving/use_manual_inputs", cfg_.use_manual_inputs);

    // RANSAC Parameters
    this->get_parameter("autonomous_driving/ransac_max_iterations", cfg_.ransac_max_iterations);
    this->get_parameter("autonomous_driving/ransac_inlier_threshold", cfg_.ransac_inlier_threshold);
    this->get_parameter("autonomous_driving/ransac_min_inlier_ratio", cfg_.ransac_min_inlier_ratio);

    // ROI Parameters
    this->get_parameter("autonomous_driving/roi_front", cfg_.param_m_ROIFront_param);
    this->get_parameter("autonomous_driving/roi_rear", cfg_.param_m_ROIRear_param);
    this->get_parameter("autonomous_driving/roi_left", cfg_.param_m_ROILeft_param);
    this->get_parameter("autonomous_driving/roi_right", cfg_.param_m_ROIRight_param);
}

void PerceptionNode::Run() {
    //===================================================
    // Get subscribe variables 
    // 일종의 input데이터 수집 단계 (멤버 변수 -> 지역변수로 복사 (mutex로 보호))
    //===================================================
    if (cfg_.use_manual_inputs == true) {
        if (b_is_manual_input_ == false) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Manual Input ...");
            return;
        }
    }

    if (b_is_simulator_on_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Vehicle State ...");
        return;
    }

    if (b_is_lane_points_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Lane Points ...");
        return;
    }

    interface::VehicleCommand manual_input; {
        if (cfg_.use_manual_inputs == true) {
            std::lock_guard<std::mutex> lock(mutex_manual_input_);
            manual_input = i_manual_input_;
        }
    }

    interface::VehicleState vehicle_state; {
        std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
        vehicle_state = i_vehicle_state_;
    }

    interface::Lane lane_points; {
        std::lock_guard<std::mutex> lock(mutex_lane_points_);
        lane_points = i_lane_points_;
    }

    //===================================================
    // Algorithm
    //===================================================
    interface::PolyfitLanes poly_lanes = FindLanes(lane_points);
    interface::PolyfitLane driving_way = FindDrivingWay(vehicle_state, poly_lanes);

    //===================================================
    // Publish output
    //===================================================
    // driving way
    p_driving_way_->publish(ros2_bridge::UpdatePolyfitLane(driving_way));
    p_poly_lanes_->publish(ros2_bridge::UpdatePolyfitLanes(poly_lanes));

}




interface::PolyfitLanes PerceptionNode::FindLanes(const interface::Lane& lane_points) {
    interface::PolyfitLanes lanes;
    lanes.frame_id = lane_points.frame_id;

    if (lane_points.point.empty()) {
        return lanes;
    }

    constexpr int kLaneCount = 4;
    const double slice_width = 0.5;  // Δx
    const int min_points_for_fit = 4;
    const int kmeans_max_iter = 10;

    double min_x = lane_points.point.front().x;
    double max_x = lane_points.point.front().x;
    for (const auto& pt : lane_points.point) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
    }

    if (max_x - min_x < slice_width) {
        max_x = min_x + slice_width;
    }

    int slice_count = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / slice_width)));
    std::vector<std::vector<interface::Point2D>> slices(static_cast<size_t>(slice_count));
    for (const auto& pt : lane_points.point) {
        int idx = static_cast<int>(std::floor((pt.x - min_x) / slice_width));
        idx = std::max(0, std::min(idx, slice_count - 1));
        slices[static_cast<size_t>(idx)].push_back(pt);
    }

    auto sliceXCenter = [&](int slice_idx, const std::vector<interface::Point2D>& pts) {
        if (pts.empty()) {
            return min_x + (slice_idx + 0.5) * slice_width;
        }
        double sum = 0.0;
        for (const auto& p : pts) {
            sum += p.x;
        }
        return sum / pts.size();
    };

    auto kmeans1D = [&](const std::vector<double>& values) {
        std::vector<std::pair<double, std::vector<double>>> clusters;
        if (values.empty()) {
            return clusters;
        }

        int k = std::min(kLaneCount, static_cast<int>(values.size()));
        std::vector<double> centers;
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        centers.reserve(k);
        for (int i = 0; i < k; ++i) {
            size_t idx = static_cast<size_t>((sorted.size() - 1) * (2 * i + 1) / (2 * k));
            centers.push_back(sorted[idx]);
        }

        std::vector<std::vector<double>> grouped(static_cast<size_t>(k));
        for (int iter = 0; iter < kmeans_max_iter; ++iter) {
            std::vector<std::vector<double>> new_grouped(static_cast<size_t>(k));
            for (double v : values) {
                int nearest = 0;
                double best = std::abs(v - centers[0]);
                for (int c = 1; c < k; ++c) {
                    double dist = std::abs(v - centers[c]);
                    if (dist < best) {
                        best = dist;
                        nearest = c;
                    }
                }
                new_grouped[static_cast<size_t>(nearest)].push_back(v);
            }

            bool converged = true;
            for (int c = 0; c < k; ++c) {
                if (new_grouped[static_cast<size_t>(c)].empty()) {
                    continue;
                }
                double mean = std::accumulate(new_grouped[static_cast<size_t>(c)].begin(), new_grouped[static_cast<size_t>(c)].end(), 0.0) /
                              new_grouped[static_cast<size_t>(c)].size();
                if (std::abs(mean - centers[c]) > 1e-3) {
                    converged = false;
                }
                centers[c] = mean;
            }
            grouped.swap(new_grouped);
            if (converged) {
                break;
            }
        }

        clusters.reserve(static_cast<size_t>(k));
        for (int c = 0; c < k; ++c) {
            clusters.push_back({centers[c], grouped[static_cast<size_t>(c)]});
        }
        std::sort(clusters.begin(), clusters.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        return clusters;
    };

    std::vector<std::vector<Eigen::Vector2d>> lane_centers(kLaneCount);
    for (size_t slice_idx = 0; slice_idx < slices.size(); ++slice_idx) {
        if (slices[slice_idx].empty()) {
            continue;
        }
        std::vector<double> ys;
        ys.reserve(slices[slice_idx].size());
        for (const auto& pt : slices[slice_idx]) {
            ys.push_back(pt.y);
        }

        auto clusters = kmeans1D(ys);
        double x_center = sliceXCenter(static_cast<int>(slice_idx), slices[slice_idx]);

        int lane_id = 0;
        for (const auto& cluster : clusters) {
            if (lane_id >= kLaneCount) {
                break;
            }
            if (cluster.second.empty()) {
                ++lane_id;
                continue;
            }
            double y_center = std::accumulate(cluster.second.begin(), cluster.second.end(), 0.0) / cluster.second.size();
            lane_centers[lane_id].push_back(Eigen::Vector2d(x_center, y_center));
            ++lane_id;
        }
    }

    auto solvePolynomial = [](const std::vector<Eigen::Vector2d>& pts) {
        Eigen::Vector4d coeffs = Eigen::Vector4d::Zero();
        if (pts.empty()) {
            return coeffs;
        }
        Eigen::MatrixXd X(pts.size(), 4);
        Eigen::VectorXd Y(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            double x = pts[i].x();
            X(static_cast<int>(i), 0) = 1.0;
            X(static_cast<int>(i), 1) = x;
            X(static_cast<int>(i), 2) = x * x;
            X(static_cast<int>(i), 3) = x * x * x;
            Y(static_cast<int>(i)) = pts[i].y();
        }
        coeffs = X.colPivHouseholderQr().solve(Y);
        return coeffs;
    };

    auto fitWithRansac = [&](const std::vector<Eigen::Vector2d>& pts) {
        if (pts.size() < static_cast<size_t>(min_points_for_fit)) {
            return solvePolynomial(pts);
        }
        std::mt19937 gen(std::random_device{}());
        Eigen::Vector4d best_coeffs = Eigen::Vector4d::Zero();
        int best_inliers = 0;
        std::vector<int> best_inlier_indices;

        std::uniform_int_distribution<> dis(0, static_cast<int>(pts.size()) - 1);
        for (int iter = 0; iter < cfg_.ransac_max_iterations; ++iter) {
            std::set<int> sample_indices;
            while (static_cast<int>(sample_indices.size()) < min_points_for_fit) {
                sample_indices.insert(dis(gen));
            }

            Eigen::MatrixXd X_sample(min_points_for_fit, 4);
            Eigen::VectorXd Y_sample(min_points_for_fit);
            int idx = 0;
            for (int sample_idx : sample_indices) {
                double x = pts[static_cast<size_t>(sample_idx)].x();
                X_sample(idx, 0) = 1.0;
                X_sample(idx, 1) = x;
                X_sample(idx, 2) = x * x;
                X_sample(idx, 3) = x * x * x;
                Y_sample(idx) = pts[static_cast<size_t>(sample_idx)].y();
                ++idx;
            }

            Eigen::Vector4d coeffs = X_sample.colPivHouseholderQr().solve(Y_sample);

            std::vector<int> inliers;
            for (size_t i = 0; i < pts.size(); ++i) {
                double x = pts[i].x();
                double y_pred = coeffs(0) + coeffs(1) * x + coeffs(2) * x * x + coeffs(3) * x * x * x;
                double error = std::abs(y_pred - pts[i].y());
                if (error < cfg_.ransac_inlier_threshold) {
                    inliers.push_back(static_cast<int>(i));
                }
            }

            if (static_cast<int>(inliers.size()) > best_inliers) {
                best_inliers = static_cast<int>(inliers.size());
                best_coeffs = coeffs;
                best_inlier_indices = inliers;
            }

            if (best_inliers > static_cast<int>(pts.size() * cfg_.ransac_min_inlier_ratio)) {
                break;
            }
        }

        if (best_inliers >= min_points_for_fit && !best_inlier_indices.empty()) {
            std::vector<Eigen::Vector2d> inlier_points;
            inlier_points.reserve(best_inlier_indices.size());
            for (int i : best_inlier_indices) {
                inlier_points.push_back(pts[static_cast<size_t>(i)]);
            }
            return solvePolynomial(inlier_points);
        }

        return solvePolynomial(pts);
    };

    for (int lane_id = 0; lane_id < kLaneCount; ++lane_id) {
        if (lane_centers[lane_id].empty()) {
            continue;
        }
        Eigen::Vector4d coeffs = fitWithRansac(lane_centers[lane_id]);
        interface::PolyfitLane lane_poly;
        lane_poly.frame_id = lane_points.frame_id;
        lane_poly.id = "lane" + std::to_string(lane_id);
        lane_poly.a0 = coeffs(0);
        lane_poly.a1 = coeffs(1);
        lane_poly.a2 = coeffs(2);
        lane_poly.a3 = coeffs(3);
        lanes.polyfitlanes.push_back(lane_poly);
    }

    return lanes;
}

interface::PolyfitLane PerceptionNode::FindDrivingWay(const interface::VehicleState &vehicle_state, const interface::PolyfitLanes& lanes) {
    (void)vehicle_state;
    interface::PolyfitLane driving_way;
    driving_way.frame_id = lanes.frame_id;

    if (lanes.polyfitlanes.empty()) {
        return driving_way;
    }

    double closest_left_dist = std::numeric_limits<double>::max();
    double closest_right_dist = std::numeric_limits<double>::max();
    Eigen::Vector4d left_coeffs = Eigen::Vector4d::Zero();
    Eigen::Vector4d right_coeffs = Eigen::Vector4d::Zero();
    bool has_left = false;
    bool has_right = false;

    for (const auto& lane : lanes.polyfitlanes) {
        Eigen::Vector4d coeff(lane.a0, lane.a1, lane.a2, lane.a3);
        double intercept = coeff(0);
        double dist = std::abs(intercept);
        if (intercept > 0.0 && dist < closest_left_dist) {
            closest_left_dist = dist;
            left_coeffs = coeff;
            has_left = true;
        } else if (intercept < 0.0 && dist < closest_right_dist) {
            closest_right_dist = dist;
            right_coeffs = coeff;
            has_right = true;
        }
    }

    if (!(has_left && has_right)) {
        return driving_way;
    }

    Eigen::Vector4d center_coeffs = (left_coeffs + right_coeffs) * 0.5;
    driving_way.id = "driving_way";
    driving_way.a0 = center_coeffs(0);
    driving_way.a1 = center_coeffs(1);
    driving_way.a2 = center_coeffs(2);
    driving_way.a3 = center_coeffs(3);
    return driving_way;
}

interface::PolyfitLane PerceptionNode::FindDrivingWayNew(const interface::VehicleState &vehicle_state, const interface::Lane& lane_points) {
    auto lanes = FindLanes(lane_points);
    auto driving_way = FindDrivingWay(vehicle_state, lanes);
    return driving_way;
}


int main(int argc, char **argv) {
    std::string node_name = "perception_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PerceptionNode>(node_name));
    rclcpp::shutdown();
    return 0;
}
