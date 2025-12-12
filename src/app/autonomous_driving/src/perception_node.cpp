/*
 * perception_node.cpp
 */
#include "autonomous_driving_config.hpp"
#include "perception_node.hpp"
#include <algorithm>
#include <limits>
#include <map>

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

    ProcessParams();

    RCLCPP_INFO(this->get_logger(), "vehicle_namespace: %s", cfg_.vehicle_namespace.c_str());
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", cfg_.loop_rate_hz);
    RCLCPP_INFO(this->get_logger(), "use_manual_inputs: %d", cfg_.use_manual_inputs);

    //===========subscriber init===============

    //(1) s_vehicle_state_
    s_vehicle_state_ = 
    this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&PerceptionNode::CallbackVehicleState, this, std::placeholders::_1));

    //(2) s_lane_points_
    s_lane_points_ =
    this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&PerceptionNode::CallbackLanePoints, this, std::placeholders::_1));


    //===========publisher init===============

    p_driving_way_ = 
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
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("autonomous_driving/use_manual_inputs", cfg_.use_manual_inputs);
    ////////////////////// TODO //////////////////////
}

void PerceptionNode::Run() {
    //===================================================
    // Get subscribe variables 
    // 일종의 input데이터 수집 단계 (멤버 변수 -> 지역변수로 복사 (mutex로 보호))
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

    // (1) Find Polyfit Lanes
    interface::PolyfitLanes poly_lanes = FindLanes(lane_points);

    // (2) Find Driving Way
    interface::PolyfitLane driving_way = FindDrivingWay(poly_lanes);

    //===================================================
    // Publish output
    //===================================================

    // (1) Publish Driving Way
    p_driving_way_->publish(ros2_bridge::UpdatePolyfitLane(driving_way));

    // (2) Publish Polyfit Lanes
    p_poly_lanes_->publish(ros2_bridge::UpdatePolyfitLanes(poly_lanes));

}

interface::PolyfitLanes PerceptionNode::FindLanes(const interface::Lane& lane_points) {
    
    // Step 0. 사용할 변수들 초기화
    interface::PolyfitLanes poly_lanes_;  // 최정 결과(lane fitting) 저장
    poly_lanes_.frame_id = lane_points.frame_id;    

    if (lane_points.point.empty()) {      // 입력으로 들어오는 점이 아무것도 없으면 fitting 결과도 없이 내보내기
        return poly_lanes_;
    }

    // Step 1. x값을 기준으로 슬라이스 나누기
    std::map<int, std::vector<interface::Point2D>> slices = SliceByX(lane_points);

    // Step 2. 각 슬라이스 별로 차선 클러스터 찾기 
    std::map<int, std::vector<PerceptionNode::Cluster>> clusters_by_slice = ClusterLanePoints(slices);
 
    if (clusters_by_slice.empty()) {     // 클러스터링이 아무것도 안되었다면 fitting 결과도 없이 내보내기
        return poly_lanes_;
    }

    // Step 3. 슬라이스 별로 차선 클러스터를 이어주기 
    // 3-1. ego에 가장 가까운 슬라이스 선택
    std::vector<int> slice_indices;
    for (const auto& kv : clusters_by_slice) slice_indices.push_back(kv.first); // 슬라이스 인덱스들을 따로 배열에 저장

    int start_idx = slice_indices.front();                  
    double best_dist = std::abs(SliceCenter(start_idx)); 
    for (int idx : slice_indices) {                         
        double dist = std::abs(SliceCenter(idx));
        if (dist < best_dist) {       // 자차 기준 슬라이스 중심 거리를 계산해서 가장 가까운 슬라이스 인덱스를 찾는다.
            best_dist = dist;   
            start_idx = idx;
        }
    }

    auto& start_clusters = clusters_by_slice[start_idx];  // 자차에 가까운 slice를 시작점으로 지정
    Cluster* left_cluster = nullptr;                      // 왼쪽 차선 
    Cluster* right_cluster = nullptr;                     // 오른쪽 차선 

    auto eval_lane = [](const interface::PolyfitLane& lane, double x) { // function:찾은 계수로 x값으로 y값 위치 계산
        return lane.a0 + lane.a1 * x + lane.a2 * x * x + lane.a3 * x * x * x;
    };

    // 3-2. 슬라이스 범위 내에서 최초 씨드를 찾기 위한 탐색 범위 (앞/뒤 N슬라이스)
    auto select_start_cluster = [&](bool is_left, double gate_center) -> Cluster* {
        Cluster* best_gate = nullptr;
        double best_gate_diff = std::numeric_limits<double>::max();
        Cluster* best_fallback = nullptr;
        double best_fallback_metric = std::numeric_limits<double>::max();

        int start_pos_int = static_cast<int>(start_idx);
        // start_idx 주변 ±start_search_span 슬라이스에서 씨드 탐색
        for (int offset = -start_search_span; offset <= start_search_span; ++offset) {
            int idx = start_pos_int + offset;
            if (idx < slice_indices.front() || idx > slice_indices.back()) continue;
            auto it = clusters_by_slice.find(idx);
            if (it == clusters_by_slice.end()) continue;
            double x_center = SliceCenter(idx);
            double gate = std::isfinite(gate_center) ? gate_center : std::numeric_limits<double>::quiet_NaN();

            for (auto& cluster : it->second) {
                bool sign_ok = is_left ? (cluster.mean_y > 0.0) : (cluster.mean_y < 0.0);
                if (!sign_ok) continue;

                double diff = std::isfinite(gate) ? std::abs(cluster.mean_y - gate) : std::numeric_limits<double>::max();
                if (std::isfinite(gate) && diff <= gate_width && diff < best_gate_diff) {
                    best_gate_diff = diff;
                    best_gate = &cluster;
                }

                double metric = is_left ? cluster.mean_y : -cluster.mean_y;
                if (metric < best_fallback_metric) {
                    best_fallback_metric = metric;
                    best_fallback = &cluster;
                }
            }

            if (best_gate != nullptr) {
                // 게이트 안에서 이미 찾았으면 추가 탐색 없이 반환
                break;
            }
        }

        return (best_gate != nullptr) ? best_gate : best_fallback;
    };

    double start_x_center = SliceCenter(start_idx);
    double left_gate_center = (has_prev_left_lane_) ? eval_lane(prev_left_lane_, start_x_center) : std::numeric_limits<double>::quiet_NaN();
    double right_gate_center = (has_prev_right_lane_) ? eval_lane(prev_right_lane_, start_x_center) : std::numeric_limits<double>::quiet_NaN();

    left_cluster = select_start_cluster(true, left_gate_center);
    right_cluster = select_start_cluster(false, right_gate_center);

    std::vector<interface::Point2D> left_points;
    std::vector<interface::Point2D> right_points;
    double left_target_y = 0.0;
    double right_target_y = 0.0;

    if (left_cluster != nullptr) {
        left_points.insert(left_points.end(), left_cluster->points.begin(), left_cluster->points.end());
        left_target_y = left_cluster->mean_y;
    }
    if (right_cluster != nullptr) {
        right_points.insert(right_points.end(), right_cluster->points.begin(), right_cluster->points.end());
        right_target_y = right_cluster->mean_y;
    }

    auto start_pos_it = std::find(slice_indices.begin(), slice_indices.end(), start_idx);
    size_t start_pos = std::distance(slice_indices.begin(), start_pos_it);

    double left_target_forward = left_target_y;
    double right_target_forward = right_target_y;
    // 이후 슬라이스에서 y가 가장 가까운 클러스터를 추적 (앞쪽)
    for (size_t i = start_pos + 1; i < slice_indices.size(); ++i) {
        int idx = slice_indices[i];
        auto& clusters = clusters_by_slice[idx];
        double x_center = SliceCenter(idx);
        double left_gate = (has_prev_left_lane_) ? eval_lane(prev_left_lane_, x_center) : std::numeric_limits<double>::quiet_NaN();
        double right_gate = (has_prev_right_lane_) ? eval_lane(prev_right_lane_, x_center) : std::numeric_limits<double>::quiet_NaN();

        if (left_cluster != nullptr) {
            const Cluster* best = nullptr;
            double best_diff = std::numeric_limits<double>::max();
            for (const auto& cluster : clusters) {
                if (cluster.mean_y <= 0.0) continue;
                double diff_target = std::abs(cluster.mean_y - left_target_forward);
                double diff_gate = std::isfinite(left_gate) ? std::abs(cluster.mean_y - left_gate) : diff_target;
                bool pass_gate = std::isfinite(left_gate) ? (diff_gate <= gate_width) : true;

                double metric = pass_gate ? diff_gate : diff_target;
                if (metric < best_diff) {
                    best_diff = metric;
                    best = &cluster;
                }
            }
            if (best != nullptr) {
                left_points.insert(left_points.end(), best->points.begin(), best->points.end());
                left_target_forward = best->mean_y;
            }
        }

        if (right_cluster != nullptr) {
            const Cluster* best = nullptr;
            double best_diff = std::numeric_limits<double>::max();
            for (const auto& cluster : clusters) {
                if (cluster.mean_y >= 0.0) continue;
                double diff_target = std::abs(cluster.mean_y - right_target_forward);
                double diff_gate = std::isfinite(right_gate) ? std::abs(cluster.mean_y - right_gate) : diff_target;
                bool pass_gate = std::isfinite(right_gate) ? (diff_gate <= gate_width) : true;

                double metric = pass_gate ? diff_gate : diff_target;
                if (metric < best_diff) {
                    best_diff = metric;
                    best = &cluster;
                }
            }
            if (best != nullptr) {
                right_points.insert(right_points.end(), best->points.begin(), best->points.end());
                right_target_forward = best->mean_y;
            }
        }
    }

    // 이전(뒤쪽) 슬라이스로도 확장 추적
    double left_target_backward = left_target_y;
    double right_target_backward = right_target_y;
    for (int i = static_cast<int>(start_pos) - 1; i >= 0; --i) {
        int idx = slice_indices[static_cast<size_t>(i)];
        auto& clusters = clusters_by_slice[idx];
        double x_center = SliceCenter(idx);
        double left_gate = (has_prev_left_lane_) ? eval_lane(prev_left_lane_, x_center) : std::numeric_limits<double>::quiet_NaN();
        double right_gate = (has_prev_right_lane_) ? eval_lane(prev_right_lane_, x_center) : std::numeric_limits<double>::quiet_NaN();

        if (left_cluster != nullptr) {
            const Cluster* best = nullptr;
            double best_diff = std::numeric_limits<double>::max();
            for (const auto& cluster : clusters) {
                if (cluster.mean_y <= 0.0) continue;
                double diff_target = std::abs(cluster.mean_y - left_target_backward);
                double diff_gate = std::isfinite(left_gate) ? std::abs(cluster.mean_y - left_gate) : diff_target;
                bool pass_gate = std::isfinite(left_gate) ? (diff_gate <= gate_width) : true;

                double metric = pass_gate ? diff_gate : diff_target;
                if (metric < best_diff) {
                    best_diff = metric;
                    best = &cluster;
                }
            }
            if (best != nullptr) {
                left_points.insert(left_points.end(), best->points.begin(), best->points.end());
                left_target_backward = best->mean_y;
            }
        }

        if (right_cluster != nullptr) {
            const Cluster* best = nullptr;
            double best_diff = std::numeric_limits<double>::max();
            for (const auto& cluster : clusters) {
                if (cluster.mean_y >= 0.0) continue;
                double diff_target = std::abs(cluster.mean_y - right_target_backward);
                double diff_gate = std::isfinite(right_gate) ? std::abs(cluster.mean_y - right_gate) : diff_target;
                bool pass_gate = std::isfinite(right_gate) ? (diff_gate <= gate_width) : true;

                double metric = pass_gate ? diff_gate : diff_target;
                if (metric < best_diff) {
                    best_diff = metric;
                    best = &cluster;
                }
            }
            if (best != nullptr) {
                right_points.insert(right_points.end(), best->points.begin(), best->points.end());
                right_target_backward = best->mean_y;
            }
        }
    }

    auto fit_lane = [&](const std::vector<interface::Point2D>& points, const std::string& id, interface::PolyfitLane& out_lane) -> bool {
        if (points.size() < 4) return false;

        Eigen::MatrixXd X(points.size(), 4);
        Eigen::VectorXd Y(points.size());

        for (size_t i = 0; i < points.size(); ++i) {
            double x = points[i].x;
            X(i, 0) = 1.0;
            X(i, 1) = x;
            X(i, 2) = x * x;
            X(i, 3) = x * x * x;
            Y(i) = points[i].y;
        }

        Eigen::VectorXd coeffs = X.colPivHouseholderQr().solve(Y);

        interface::PolyfitLane lane;
        lane.frame_id = lane_points.frame_id;
        lane.id = id;
        lane.a0 = coeffs(0);
        lane.a1 = coeffs(1);
        lane.a2 = coeffs(2);
        lane.a3 = coeffs(3);

        out_lane = lane;
        poly_lanes_.polyfitlanes.push_back(lane);
        return true;
    };

    interface::PolyfitLane fitted_left_lane;
    interface::PolyfitLane fitted_right_lane;
    bool left_fit = fit_lane(left_points, "left_lane", fitted_left_lane);
    bool right_fit = fit_lane(right_points, "right_lane", fitted_right_lane);

    if (left_fit) {
        prev_left_lane_ = fitted_left_lane;
        has_prev_left_lane_ = true;
    }
    if (right_fit) {
        prev_right_lane_ = fitted_right_lane;
        has_prev_right_lane_ = true;
    }

    return poly_lanes_;
}

std::map<int, std::vector<interface::Point2D>> PerceptionNode::SliceByX(const interface::Lane& lane_points){

    std::map<int, std::vector<interface::Point2D>> slices;

    // 1-1. 들어온 lane_points의 x 최대 최소값을 구한다
    min_x = lane_points.point.front().x;
    max_x = lane_points.point.front().x;
    for (const auto& pt : lane_points.point) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
    }

    // 1-2. 미리 설정한 슬라이스 폭으로 들어오는 lane points들을 슬라이스별로 나누어서 slice index 부여.
    for (const auto& pt : lane_points.point) {
        int slice_idx = static_cast<int>(std::floor((pt.x - min_x) / slice_width));   
        slices[slice_idx].push_back(pt);
    }

    return slices;
}

std::map<int, std::vector<Cluster>> PerceptionNode::ClusterLanePoints(std::map<int, std::vector<interface::Point2D>> slices){
    
    // 슬라이스 별로 차선이 저장된 cluster 들의 모임
    std::map<int, std::vector<Cluster>> clusters_by_slice;  

    if (hist_bin_width <= 0.0) {  // 히스토그램 bin의 너비가 0이면 
        return clusters_by_slice;
    }

    // slice 별로 차선의 Cluster 찾기
    for (const auto& entry : slices) {
        int idx = entry.first;                  // slice index
        const auto& pts = entry.second;         // slice points
        if (pts.empty()) continue;              // 현재 슬라이스의 포인트가 없다면 다음 슬라이스로 

        std::vector<interface::Point2D> sorted_pts = pts;   // 포인트를 y값을 기준으로 정렬
        std::sort(sorted_pts.begin(), sorted_pts.end(), [](const auto& a, const auto& b) {
            return a.y < b.y;
        });

        // 히스토그램 준비
        double min_y = sorted_pts.front().y;
        double max_y = sorted_pts.back().y;
        int bin_count = std::max(1, static_cast<int>(std::ceil((max_y - min_y) / hist_bin_width)));
        std::vector<std::vector<size_t>> bins(bin_count);
        for (size_t i = 0; i < sorted_pts.size(); ++i) {
            int bin_idx = std::min(bin_count - 1, static_cast<int>(std::floor((sorted_pts[i].y - min_y) / hist_bin_width)));
            bins[bin_idx].push_back(i);
        }

        // 히스토그램을 훑으면서 빈 구간을 기준으로 클러스터 분리
        std::vector<Cluster> clusters;          
        Cluster cur_cluster;
        double sum_y = 0.0;
        double sum_x = 0.0;
        int empty_run = 0;

        auto add_point = [&](const interface::Point2D& pt) {
            cur_cluster.points.push_back(pt);
            sum_y += pt.y;
            sum_x += pt.x;
        };

        auto flush_cluster = [&]() {
            if (cur_cluster.points.empty()) return;
            double n = static_cast<double>(cur_cluster.points.size());
            cur_cluster.mean_y = sum_y / n;
            cur_cluster.mean_x = sum_x / n;
            clusters.push_back(cur_cluster);
            cur_cluster = Cluster{};
            sum_y = 0.0;
            sum_x = 0.0;
        };

        for (int bin_idx = 0; bin_idx < bin_count; ++bin_idx) {
            const auto& bin_points_idx = bins[bin_idx];
            if (!bin_points_idx.empty()) {
                empty_run = 0;
                for (size_t point_idx : bin_points_idx) {
                    add_point(sorted_pts[point_idx]);
                }
            } else {
                ++empty_run;
                if (empty_run > empty_bin_gap) {
                    flush_cluster();
                }
            }
        }

        flush_cluster();
        clusters_by_slice[idx] = clusters;
    }

    return clusters_by_slice;
}

interface::PolyfitLane PerceptionNode::FindDrivingWay(const interface::PolyfitLanes& poly_lanes) {
    
    interface::PolyfitLane driving_way_;
    driving_way_.frame_id = poly_lanes.frame_id;

    const interface::PolyfitLane* left_lane = nullptr;
    const interface::PolyfitLane* right_lane = nullptr;

    for (const auto& lane : poly_lanes.polyfitlanes) {
        if (lane.id == "left_lane") {
            left_lane = &lane;
        } else if (lane.id == "right_lane") {
            right_lane = &lane;
        }
    }

    bool has_candidate = false;

    // Case 1: 두 차선 모두 있는 경우, 계수 평균으로 중앙 차선 생성
    if (left_lane != nullptr && right_lane != nullptr) {
        driving_way_.id = "driving_way_center";
        driving_way_.a0 = (left_lane->a0 + right_lane->a0) * 0.5;
        driving_way_.a1 = (left_lane->a1 + right_lane->a1) * 0.5;
        driving_way_.a2 = (left_lane->a2 + right_lane->a2) * 0.5;
        driving_way_.a3 = (left_lane->a3 + right_lane->a3) * 0.5;
        has_candidate = true;
    } else if (left_lane != nullptr || right_lane != nullptr) {
        // Case 2: 한 개 차선만 있는 경우, 자차쪽으로 2 m 오프셋
        const double offset = 2.0; // [m]
        const interface::PolyfitLane* single_lane = (left_lane != nullptr) ? left_lane : right_lane;

        driving_way_.id = "driving_way_offset";
        driving_way_.a0 = single_lane->a0;
        driving_way_.a1 = single_lane->a1;
        driving_way_.a2 = single_lane->a2;
        driving_way_.a3 = single_lane->a3;

        double offset_sign = 0.0;
        if (single_lane->id == "left_lane") {
            offset_sign = -offset;
        } else if (single_lane->id == "right_lane") {
            offset_sign = offset;
        } else {
            offset_sign = (single_lane->a0 >= 0.0) ? -offset : offset;
        }
        driving_way_.a0 += offset_sign;
        has_candidate = true;
    }

    // Case 3: 폴리핏 차선 없음 -> 이전 값을 유지하거나 기본값 반환
    if (!has_candidate) {
        if (has_prev_driving_way_) {
            return prev_driving_way_;
        }
        driving_way_.id = "driving_way_unknown";
        return driving_way_;
    }

    // 계수 스무딩: 이전 프레임과 지수 가중 평균
    if (has_prev_driving_way_) {
        double alpha = driving_way_smooth_alpha_;
        driving_way_.a0 = alpha * driving_way_.a0 + (1.0 - alpha) * prev_driving_way_.a0;
        driving_way_.a1 = alpha * driving_way_.a1 + (1.0 - alpha) * prev_driving_way_.a1;
        driving_way_.a2 = alpha * driving_way_.a2 + (1.0 - alpha) * prev_driving_way_.a2;
        driving_way_.a3 = alpha * driving_way_.a3 + (1.0 - alpha) * prev_driving_way_.a3;
    }

    prev_driving_way_ = driving_way_;
    has_prev_driving_way_ = true;
    return driving_way_;
}


int main(int argc, char **argv) {
    std::string node_name = "perception_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PerceptionNode>(node_name));
    rclcpp::shutdown();
    return 0;
}
