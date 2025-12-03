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
#include <cmath>

using namespace Eigen;
using namespace std;

PerceptionNode::PerceptionNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options) {

    //QoS init 
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    //===============parameters===============
    //declare parameters(파라미터 등록+초기값 설정)
    this->declare_parameter("autonomous_driving/ns", "");
    this->declare_parameter("autonomous_driving/loop_rate_hz", 100.0);

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
    
    //===========subscriber init===============

    s_vehicle_state_ = 
    this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&PerceptionNode::CallbackVehicleState, this, std::placeholders::_1));

    s_lane_points_ = 
    this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&PerceptionNode::CallbackLanePoints, this, std::placeholders::_1));

    //===========publisher init===============

    p_driving_way_= 
    this->create_publisher<ad_msgs::msg::PolyfitLaneData>(
        "driving_way", qos_profile);

    p_poly_lanes_ = 
    this->create_publisher<ad_msgs::msg::PolyfitLaneDataArray>(
        "poly_lanes", qos_profile);
    
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); });
}

PerceptionNode::~PerceptionNode() {}

void PerceptionNode::ProcessParams() {
    //get parameters : 선언한 파라미터 값을 읽어오기
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);

    // RANSAC Parameters
    this->get_parameter("autonomous_driving/ransac_max_iterations", ransac_max_iterations);
    this->get_parameter("autonomous_driving/ransac_inlier_threshold", ransac_inlier_threshold);
    this->get_parameter("autonomous_driving/ransac_min_inlier_ratio", ransac_min_inlier_ratio);

    // ROI Parameters
    this->get_parameter("autonomous_driving/roi_front", cfg_.param_m_ROIFront_param);
    this->get_parameter("autonomous_driving/roi_rear", cfg_.param_m_ROIRear_param);
    this->get_parameter("autonomous_driving/roi_left", cfg_.param_m_ROILeft_param);
    this->get_parameter("autonomous_driving/roi_right", cfg_.param_m_ROIRight_param);
}

void PerceptionNode::Run() {
    //===================================================
    // Get subscribe variables 
    //===================================================
    if (b_is_simulator_on_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Vehicle State ...");
        return;
    }
    if (b_is_lane_points_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Lane Points ...");
        return;
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
    
    // 이전 슬라이스의 차선 Y 위치를 추적 (차선 일관성 유지용)
    std::vector<double> prev_lane_positions(kLaneCount, std::numeric_limits<double>::quiet_NaN());
    bool prev_initialized = false;
    
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

        // 비어있지 않은 클러스터만 추출
        std::vector<std::pair<double, double>> valid_clusters; // (center_y, mean_y)
        for (const auto& cluster : clusters) {
            if (!cluster.second.empty()) {
                double y_mean = std::accumulate(cluster.second.begin(), cluster.second.end(), 0.0) / cluster.second.size();
                valid_clusters.push_back({cluster.first, y_mean});
            }
        }

        if (valid_clusters.empty()) {
            continue;
        }

        // Y값 기준으로 정렬 (왼쪽 차선이 Y값이 큼 -> 내림차순 정렬)
        std::sort(valid_clusters.begin(), valid_clusters.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;  // Y값 내림차순 (좌측부터)
        });

        if (!prev_initialized) {
            // 첫 번째 유효한 슬라이스: 클러스터를 차선에 직접 할당
            // 클러스터 개수에 따라 차선 ID 배치 (중앙 차선 우선)
            int num_clusters = static_cast<int>(valid_clusters.size());
            std::vector<int> lane_assignments;
            
            if (num_clusters == 4) {
                lane_assignments = {0, 1, 2, 3};  // 모든 차선
            } else if (num_clusters == 3) {
                lane_assignments = {0, 1, 2};     // 3개 차선
            } else if (num_clusters == 2) {
                lane_assignments = {1, 2};        // 중앙 2개 차선 (ego vehicle 양옆)
            } else {
                lane_assignments = {1};           // 1개면 왼쪽 차선으로 가정
            }

            for (size_t i = 0; i < valid_clusters.size() && i < lane_assignments.size(); ++i) {
                int lane_id = lane_assignments[i];
                if (lane_id < kLaneCount) {
                    lane_centers[lane_id].push_back(Eigen::Vector2d(x_center, valid_clusters[i].second));
                    prev_lane_positions[lane_id] = valid_clusters[i].second;
                }
            }
            prev_initialized = true;
        } else {
            // 이전 슬라이스 기반 매칭: Hungarian-like greedy matching
            std::vector<bool> cluster_used(valid_clusters.size(), false);
            std::vector<bool> lane_matched(kLaneCount, false);
            
            // 각 이전 차선 위치에 가장 가까운 클러스터 매칭
            for (int lane_id = 0; lane_id < kLaneCount; ++lane_id) {
                if (std::isnan(prev_lane_positions[lane_id])) {
                    continue;
                }
                
                double min_dist = std::numeric_limits<double>::max();
                int best_cluster = -1;
                
                for (size_t c = 0; c < valid_clusters.size(); ++c) {
                    if (cluster_used[c]) {
                        continue;
                    }
                    double dist = std::abs(valid_clusters[c].second - prev_lane_positions[lane_id]);
                    // 차선 간격의 절반(약 1.75m) 이내만 매칭 허용
                    if (dist < min_dist && dist < 1.75) {
                        min_dist = dist;
                        best_cluster = static_cast<int>(c);
                    }
                }
                
                if (best_cluster >= 0) {
                    cluster_used[best_cluster] = true;
                    lane_matched[lane_id] = true;
                    lane_centers[lane_id].push_back(Eigen::Vector2d(x_center, valid_clusters[best_cluster].second));
                    prev_lane_positions[lane_id] = valid_clusters[best_cluster].second;
                }
            }
            
            // 매칭되지 않은 클러스터 처리: 새로운 차선으로 할당
            for (size_t c = 0; c < valid_clusters.size(); ++c) {
                if (cluster_used[c]) {
                    continue;
                }
                // 빈 차선 슬롯 찾기 (Y 위치 기준으로 적절한 슬롯 선택)
                double cluster_y = valid_clusters[c].second;
                int best_lane = -1;
                
                for (int lane_id = 0; lane_id < kLaneCount; ++lane_id) {
                    if (!lane_matched[lane_id]) {
                        // 아직 매칭 안 된 슬롯 중 Y 위치가 적절한 것 선택
                        if (best_lane == -1) {
                            best_lane = lane_id;
                        } else {
                            // 차선 순서 유지: lane_id가 작을수록 Y가 커야 함
                            bool should_use = false;
                            if (lane_id < best_lane && cluster_y > 0) {
                                should_use = true;
                            } else if (lane_id > best_lane && cluster_y < 0) {
                                should_use = true;
                            }
                            if (should_use) {
                                best_lane = lane_id;
                            }
                        }
                    }
                }
                
                if (best_lane >= 0 && best_lane < kLaneCount) {
                    lane_matched[best_lane] = true;
                    lane_centers[best_lane].push_back(Eigen::Vector2d(x_center, cluster_y));
                    prev_lane_positions[best_lane] = cluster_y;
                }
            }
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
        for (int iter = 0; iter < ransac_max_iterations; ++iter) {
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
                if (error < ransac_inlier_threshold) {
                    inliers.push_back(static_cast<int>(i));
                }
            }

            if (static_cast<int>(inliers.size()) > best_inliers) {
                best_inliers = static_cast<int>(inliers.size());
                best_coeffs = coeffs;
                best_inlier_indices = inliers;
            }

            if (best_inliers > static_cast<int>(pts.size() * ransac_min_inlier_ratio)) {
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

int main(int argc, char **argv) {
    std::string node_name = "perception_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PerceptionNode>(node_name));
    rclcpp::shutdown();
    return 0;
}
