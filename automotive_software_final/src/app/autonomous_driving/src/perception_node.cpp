/**
 * @file perception_node.cpp
 * @brief Robust Lane Detection Module for Autonomous Driving
 * 
 * Main Functions:
 * - FindLanes: Detect 2 closest lanes (left/right)
 * - FindDrivingWay: Generate driving path from detected lanes
 * 
 * Helper Functions (4):
 * - SliceByX: X축 기준 포인트 분할
 * - ClusterByY: Y값 기준 K-means 클러스터링  
 * - MatchClusters: 클러스터-차선 매칭 (ego-based + tracking)
 * - FitPolynomial: RANSAC 다항식 피팅
 * 
 * @note SWE.3: Software Detailed Design & Unit Construction
 */
#include "autonomous_driving_config.hpp"
#include "perception_node.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <cmath>
#include <random>

using namespace Eigen;
using namespace std;

//=============================================================================
// Constants
//=============================================================================
namespace {
    constexpr int kLaneCount = 2;
    constexpr int kMinPointsForFit = 4;
    constexpr double kSliceWidth = 0.5;
    constexpr double kMatchingThreshold = 1.75;
    constexpr int kKmeansMaxIter = 10;
    constexpr double kConvergenceThreshold = 1e-3;
    constexpr double kEmaAlpha = 0.4;
    constexpr double kLaneWidth = 3.5;
    constexpr int kRansacMaxIter = 50;
    constexpr double kRansacInlierThreshold = 0.15;
    constexpr double kRansacMinInlierRatio = 0.7;
    constexpr int kMaxFramesWithoutUpdate = 5;
}

PerceptionNode::PerceptionNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options) {

    //QoS init 
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    //===============parameters===============
    this->declare_parameter("autonomous_driving/ns", "");
    this->declare_parameter("autonomous_driving/loop_rate_hz", 100.0);

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
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
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
//=============================================================================
// Data Structures
//=============================================================================
struct PointSlice {
    double x_center;
    std::vector<interface::Point2D> points;
};

struct LaneCluster {
    double center_y;
    std::vector<double> y_values;
};

struct LaneTrack {
    std::vector<Eigen::Vector2d> points;
    double last_y;
    bool active;
};


//=============================================================================
// Helper Function 1: SliceByX
// - X축 기준으로 포인트를 슬라이스로 분할
//=============================================================================
std::vector<PointSlice> SliceByX(const interface::Lane& lane_points, double width) {
    std::vector<PointSlice> slices;
    if (lane_points.point.empty()) return slices;

    // Find X range
    double min_x = lane_points.point.front().x;
    double max_x = lane_points.point.front().x;
    for (const auto& pt : lane_points.point) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
    }
    if (max_x - min_x < width) max_x = min_x + width;

    // Create and fill slices
    int count = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / width)));
    slices.resize(count);
    
    for (int i = 0; i < count; ++i) {
        slices[i].x_center = min_x + (i + 0.5) * width;
    }

    for (const auto& pt : lane_points.point) {
        int idx = std::clamp(static_cast<int>((pt.x - min_x) / width), 0, count - 1);
        slices[idx].points.push_back(pt);
    }

    // Update centers from actual points
    for (auto& s : slices) {
        if (!s.points.empty()) {
            double sum = 0.0;
            for (const auto& p : s.points) sum += p.x;
            s.x_center = sum / s.points.size();
        }
    }

    return slices;
}


//=============================================================================
// Helper Function 2: ClusterByY
// - Y값 기준 1D K-means 클러스터링
//=============================================================================
std::vector<LaneCluster> ClusterByY(const std::vector<double>& y_vals, int max_k) {
    std::vector<LaneCluster> result;
    if (y_vals.empty()) return result;

    int k = std::min(max_k, static_cast<int>(y_vals.size()));

    // Initialize centers from quantiles
    std::vector<double> sorted = y_vals;
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> centers(k);
    for (int i = 0; i < k; ++i) {
        centers[i] = sorted[(sorted.size() - 1) * (2 * i + 1) / (2 * k)];
    }

    // K-means iteration
    std::vector<std::vector<double>> groups(k);
    for (int iter = 0; iter < kKmeansMaxIter; ++iter) {
        std::vector<std::vector<double>> new_groups(k);
        
        for (double y : y_vals) {
            int best = 0;
            double best_dist = std::abs(y - centers[0]);
            for (int c = 1; c < k; ++c) {
                double d = std::abs(y - centers[c]);
                if (d < best_dist) { best_dist = d; best = c; }
            }
            new_groups[best].push_back(y);
        }

        bool converged = true;
        for (int c = 0; c < k; ++c) {
            if (new_groups[c].empty()) continue;
            double mean = std::accumulate(new_groups[c].begin(), new_groups[c].end(), 0.0) 
                          / new_groups[c].size();
            if (std::abs(mean - centers[c]) > kConvergenceThreshold) converged = false;
            centers[c] = mean;
        }
        groups.swap(new_groups);
        if (converged) break;
    }

    // Build result
    for (int c = 0; c < k; ++c) {
        if (!groups[c].empty()) {
            result.push_back({centers[c], groups[c]});
        }
    }
    
    // Sort by Y descending
    std::sort(result.begin(), result.end(), 
              [](const auto& a, const auto& b) { return a.center_y > b.center_y; });
    
    return result;
}


//=============================================================================
// Helper Function 3: MatchClusters
// - 클러스터를 차선 트랙에 매칭
// - 첫 슬라이스: ego-based (Y>0→left, Y<0→right)
// - 이후 슬라이스: 이전 Y 위치 기반 추적
//=============================================================================
void MatchClusters(const std::vector<LaneCluster>& clusters, double x,
                   std::vector<LaneTrack>& tracks, double threshold, bool is_first) {
    
    // Initialize tracks
    if (tracks.empty()) {
        tracks.resize(kLaneCount);
        for (auto& t : tracks) { t.last_y = NAN; t.active = false; }
    }
    if (clusters.empty()) return;

    std::vector<bool> used(clusters.size(), false);

    if (is_first) {
        // === Ego-based selection for first slice ===
        // Left: Y>0 중 가장 작은 값 (자차에 가장 가까운 왼쪽)
        // Right: Y<0 중 가장 큰 값 (자차에 가장 가까운 오른쪽)
        int left_idx = -1, right_idx = -1;
        double closest_left = std::numeric_limits<double>::max();
        double closest_right = std::numeric_limits<double>::lowest();

        for (size_t c = 0; c < clusters.size(); ++c) {
            double y = clusters[c].center_y;
            if (y > 0.0 && y < closest_left) {
                closest_left = y; left_idx = c;
            } else if (y <= 0.0 && y > closest_right) {
                closest_right = y; right_idx = c;
            }
        }

        if (left_idx >= 0) {
            double y = clusters[left_idx].center_y;
            tracks[0].points.push_back({x, y});
            tracks[0].last_y = y;
            tracks[0].active = true;
            used[left_idx] = true;
        }
        if (right_idx >= 0) {
            double y = clusters[right_idx].center_y;
            tracks[1].points.push_back({x, y});
            tracks[1].last_y = y;
            tracks[1].active = true;
            used[right_idx] = true;
        }
    } else {
        // === Tracking by previous Y position ===
        for (int lane = 0; lane < kLaneCount; ++lane) {
            if (!tracks[lane].active || std::isnan(tracks[lane].last_y)) continue;
            
            int best = -1;
            double min_dist = threshold;
            for (size_t c = 0; c < clusters.size(); ++c) {
                if (used[c]) continue;
                double d = std::abs(clusters[c].center_y - tracks[lane].last_y);
                if (d < min_dist) { min_dist = d; best = c; }
            }
            
            if (best >= 0) {
                double y = clusters[best].center_y;
                tracks[lane].points.push_back({x, y});
                tracks[lane].last_y = y;
                used[best] = true;
            }
        }
    }
}


//=============================================================================
// Helper Function 4: FitPolynomial
// - RANSAC 기반 3차 다항식 피팅 (노이즈 강건)
// - 포인트 부족 시 LSM 사용
//=============================================================================
Eigen::Vector4d FitPolynomial(const std::vector<Eigen::Vector2d>& pts) {
    Eigen::Vector4d coeffs = Eigen::Vector4d::Zero();
    if (pts.empty()) return coeffs;

    // LSM fitting lambda
    auto fit_lsm = [](const std::vector<Eigen::Vector2d>& p) {
        Eigen::MatrixXd X(p.size(), 4);
        Eigen::VectorXd Y(p.size());
        for (size_t i = 0; i < p.size(); ++i) {
            double x = p[i].x();
            X(i, 0) = 1.0; X(i, 1) = x; X(i, 2) = x*x; X(i, 3) = x*x*x;
            Y(i) = p[i].y();
        }
        return X.colPivHouseholderQr().solve(Y);
    };

    // Not enough points for RANSAC
    if (pts.size() < 4) return fit_lsm(pts);

    // RANSAC
    Eigen::Vector4d best = Eigen::Vector4d::Zero();
    size_t best_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> indices(pts.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int iter = 0; iter < kRansacMaxIter; ++iter) {
        std::shuffle(indices.begin(), indices.end(), gen);

        // Fit with 4 random points
        std::vector<Eigen::Vector2d> sample(4);
        for (int i = 0; i < 4; ++i) sample[i] = pts[indices[i]];
        Eigen::Vector4d c = fit_lsm(sample);

        // Count inliers
        std::vector<Eigen::Vector2d> inliers;
        for (const auto& pt : pts) {
            double x = pt.x();
            double y_pred = c(0) + c(1)*x + c(2)*x*x + c(3)*x*x*x;
            if (std::abs(pt.y() - y_pred) < kRansacInlierThreshold) {
                inliers.push_back(pt);
            }
        }

        if (inliers.size() > best_count) {
            best_count = inliers.size();
            best = fit_lsm(inliers);
        }

        if (static_cast<double>(best_count) / pts.size() >= kRansacMinInlierRatio) break;
    }

    return best;
}


//=============================================================================
// Main Function 1: FindLanes
//=============================================================================
interface::PolyfitLanes PerceptionNode::FindLanes(const interface::Lane& lane_points) {
    interface::PolyfitLanes lanes;
    lanes.frame_id = lane_points.frame_id;

    // Handle empty input - use previous
    if (lane_points.point.empty()) {
        for (int i = 0; i < kLaneCount; ++i) {
            if (prev_lane_valid_[i] && prev_frames_without_update_[i] < kMaxFramesWithoutUpdate) {
                interface::PolyfitLane lane;
                lane.frame_id = lane_points.frame_id;
                lane.id = (i == 0) ? "left_lane" : "right_lane";
                lane.a0 = prev_lane_coeffs_[i](0);
                lane.a1 = prev_lane_coeffs_[i](1);
                lane.a2 = prev_lane_coeffs_[i](2);
                lane.a3 = prev_lane_coeffs_[i](3);
                lanes.polyfitlanes.push_back(lane);
                prev_frames_without_update_[i]++;
            }
        }
        return lanes;
    }

    // Step 1: Slice by X
    auto slices = SliceByX(lane_points, kSliceWidth);
    std::sort(slices.begin(), slices.end(), 
              [](const auto& a, const auto& b) { return a.x_center < b.x_center; });

    // Step 2: Cluster and match
    std::vector<LaneTrack> tracks;
    bool first = true;
    
    for (const auto& slice : slices) {
        if (slice.points.empty()) continue;
        
        std::vector<double> y_vals;
        for (const auto& pt : slice.points) y_vals.push_back(pt.y);
        
        auto clusters = ClusterByY(y_vals, 4);
        if (clusters.empty()) continue;
        
        MatchClusters(clusters, slice.x_center, tracks, kMatchingThreshold, first);
        if (first && !tracks.empty()) first = false;
    }

    // Step 3: Fit polynomials
    bool detected[kLaneCount] = {false, false};
    Eigen::Vector4d coeffs[kLaneCount];

    for (int i = 0; i < kLaneCount; ++i) {
        if (i >= static_cast<int>(tracks.size())) continue;
        
        if (tracks[i].points.size() >= kMinPointsForFit) {
            Eigen::Vector4d c = FitPolynomial(tracks[i].points);
            
            // EMA smoothing
            if (prev_lane_valid_[i]) {
                c = kEmaAlpha * c + (1.0 - kEmaAlpha) * prev_lane_coeffs_[i];
            }
            
            coeffs[i] = c;
            detected[i] = true;
            prev_lane_coeffs_[i] = c;
            prev_lane_valid_[i] = true;
            prev_frames_without_update_[i] = 0;
            
        } else if (prev_lane_valid_[i] && prev_frames_without_update_[i] < kMaxFramesWithoutUpdate) {
            coeffs[i] = prev_lane_coeffs_[i];
            detected[i] = true;
            prev_frames_without_update_[i]++;
        }
    }

    // Step 4: Recover missing lane from neighbor
    if (detected[0] && !detected[1]) {
        coeffs[1] = coeffs[0];
        coeffs[1](0) -= kLaneWidth;
        detected[1] = true;
    } else if (!detected[0] && detected[1]) {
        coeffs[0] = coeffs[1];
        coeffs[0](0) += kLaneWidth;
        detected[0] = true;
    }

    // Step 5: Build output
    for (int i = 0; i < kLaneCount; ++i) {
        if (!detected[i]) continue;
        interface::PolyfitLane lane;
        lane.frame_id = lane_points.frame_id;
        lane.id = (i == 0) ? "left_lane" : "right_lane";
        lane.a0 = coeffs[i](0);
        lane.a1 = coeffs[i](1);
        lane.a2 = coeffs[i](2);
        lane.a3 = coeffs[i](3);
        lanes.polyfitlanes.push_back(lane);
    }

    return lanes;
}


//=============================================================================
// Main Function 2: FindDrivingWay
//=============================================================================
interface::PolyfitLane PerceptionNode::FindDrivingWay(
    const interface::VehicleState& vehicle_state, 
    const interface::PolyfitLanes& lanes) {
    
    (void)vehicle_state;
    
    interface::PolyfitLane driving_way;
    driving_way.frame_id = lanes.frame_id;
    driving_way.id = "driving_way";

    Eigen::Vector4d left = Eigen::Vector4d::Zero();
    Eigen::Vector4d right = Eigen::Vector4d::Zero();
    bool has_left = false, has_right = false;

    for (const auto& lane : lanes.polyfitlanes) {
        if (lane.id == "left_lane") {
            left = Eigen::Vector4d(lane.a0, lane.a1, lane.a2, lane.a3);
            has_left = true;
        } else if (lane.id == "right_lane") {
            right = Eigen::Vector4d(lane.a0, lane.a1, lane.a2, lane.a3);
            has_right = true;
        }
    }

    Eigen::Vector4d center;
    
    if (has_left && has_right) {
        center = (left + right) * 0.5;
        prev_driving_way_coeffs_ = center;
        prev_driving_way_valid_ = true;
        prev_driving_way_frames_ = 0;
    } else if (has_left) {
        center = left;
        center(0) -= kLaneWidth * 0.5;
        prev_driving_way_coeffs_ = center;
        prev_driving_way_valid_ = true;
        prev_driving_way_frames_ = 0;
    } else if (has_right) {
        center = right;
        center(0) += kLaneWidth * 0.5;
        prev_driving_way_coeffs_ = center;
        prev_driving_way_valid_ = true;
        prev_driving_way_frames_ = 0;
    } else {
        if (prev_driving_way_valid_ && prev_driving_way_frames_ < kMaxFramesWithoutUpdate) {
            center = prev_driving_way_coeffs_;
            prev_driving_way_frames_++;
        } else {
            return driving_way;  // Zero coefficients
        }
    }

    driving_way.a0 = center(0);
    driving_way.a1 = center(1);
    driving_way.a2 = center(2);
    driving_way.a3 = center(3);
    
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