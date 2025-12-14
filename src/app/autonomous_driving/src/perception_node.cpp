/*
 * perception_node.cpp
 * export LIBGL_ALWAYS_SOFTWARE=1
 */
#include "autonomous_driving_config.hpp"
#include "perception_node.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <cctype>

using namespace Eigen;
using namespace std;

PerceptionNode::PerceptionNode(const std::string &node_name, const rclcpp::NodeOptions &options) : Node(node_name, options) {

    //QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    //===============parameters===============
    //declare parameters(파라미터 등록+초기값 설정)
    this->declare_parameter("autonomous_driving/ns", "");
    this->declare_parameter("autonomous_driving/loop_rate_hz", 100.0);
    this->declare_parameter("perception/slice_width", slice_width);
    this->declare_parameter("perception/cluster_threshold", cluster_threshold);
    this->declare_parameter("perception/gate_width", gate_width);
    this->declare_parameter("perception/hist_bin_width_scale", 0.25); // hist_bin_width = cluster_threshold * scale
    this->declare_parameter("perception/side_lane_window", side_lane_window);
    this->declare_parameter("perception/lane_disconnect_gap", lane_disconnect_gap);

    ProcessParams();

    RCLCPP_INFO(this->get_logger(), "vehicle_namespace: %s", cfg_.vehicle_namespace.c_str());
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", cfg_.loop_rate_hz);

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
    this->get_parameter("perception/slice_width", slice_width);
    this->get_parameter("perception/cluster_threshold", cluster_threshold);
    this->get_parameter("perception/gate_width", gate_width);
    double hist_scale = 0.25;
    this->get_parameter("perception/hist_bin_width_scale", hist_scale);
    hist_bin_width = cluster_threshold * hist_scale;
    this->get_parameter("perception/side_lane_window", side_lane_window);
    this->get_parameter("perception/lane_disconnect_gap", lane_disconnect_gap);
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
    
    // ----------------------------------------------------------------------------------
    // STEP 0. 사용할 변수들 초기화
    // ----------------------------------------------------------------------------------

    interface::PolyfitLanes poly_lanes_;  // 최정 결과(lane fitting) 저장
    poly_lanes_.frame_id = lane_points.frame_id;    

    if (lane_points.point.empty()) {      // 입력으로 들어오는 점이 아무것도 없으면 fitting 결과도 없이 내보내기
        return poly_lanes_;
    }

    // ----------------------------------------------------------------------------------
    // STEP 1. X 축 방향으로 슬라이스 나누기
    // ----------------------------------------------------------------------------------

    std::map<int, std::vector<interface::Point2D>> slices = SliceByX(lane_points);

    // ----------------------------------------------------------------------------------
    // STEP 2. 각 슬라이스 별로 차선 클러스터 찾기
    // ----------------------------------------------------------------------------------

    std::map<int, std::vector<PerceptionNode::Cluster>> clusters_by_slice = FindCluster(slices); // 슬라이스 인덱스 별로 찾은 클러스터들 저장
 
    if (clusters_by_slice.empty()) {     // 클러스터링이 아무것도 안되었다면 fitting 결과도 없이 내보내기
        return poly_lanes_;
    }

    // ----------------------------------------------------------------------------------
    // STEP 3. 슬라이스 순서대로 클러스터 추적 (lane_id 0~3 관리)
    // ----------------------------------------------------------------------------------

    auto IsLeft = [&](double y) { return y > 0.1; };   // 좌측(y+) 허용 오차
    auto IsRight = [&](double y) { return y < -0.1; }; // 우측(y-) 허용 오차

    auto get_gate_center = [&](int lane_id, double x) -> double {
        if (lane_id < 1 || lane_id > 2) return std::numeric_limits<double>::quiet_NaN();
        int idx = lane_id - 1; // 1->0, 2->1
        if (has_prev_lane_id_[idx]) {
            return EvalLane(prev_lane_id_[idx], x);
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    struct TrackState {
        bool active{false};
        double last_y{0.0};
        std::vector<interface::Point2D> pts;
    };
    TrackState tracks[2]; // 0: lane1(left), 1: lane2(right)
    for (int i = 0; i < 2; ++i) {
        if (has_prev_lane_id_[i]) {
            tracks[i].active = true;
            tracks[i].last_y = EvalLane(prev_lane_id_[i], 0.0);
        }
    }

    std::vector<int> slice_index_array;
    for (const auto& kv : clusters_by_slice) slice_index_array.push_back(kv.first);

    for (int idx : slice_index_array) {
        auto& clusters = clusters_by_slice[idx];
        double x_center = SliceCenter(idx);

        double gate_y[2];
        for (int i = 0; i < 2; ++i) gate_y[i] = get_gate_center(i + 1, x_center);

        struct Candidate {
            double cost;
            int track;
            int cluster;
        };
        std::vector<Candidate> cands;
        cands.reserve(clusters.size() * 2);

        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            const auto& cl = clusters[ci];
            for (int ti = 0; ti < 2; ++ti) {
                bool track_left = (ti == 0);
                if (track_left && !IsLeft(cl.mean_y)) continue;
                if (!track_left && !IsRight(cl.mean_y)) continue;
                double gate = gate_y[ti];
                double cost = std::numeric_limits<double>::max();
                if (std::isfinite(gate)) {
                    double diff = std::abs(cl.mean_y - gate);
                    if (diff > gate_width) continue;
                    cost = diff;
                } else if (tracks[ti].active) {
                    cost = std::abs(cl.mean_y - tracks[ti].last_y);
                } else {
                    cost = std::abs(cl.mean_y);
                }
                cands.push_back({cost, ti, static_cast<int>(ci)});
            }
        }

        std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
            return a.cost < b.cost;
        });

        std::vector<bool> cluster_used(clusters.size(), false);
        bool track_used[2] = {false, false};

        for (const auto& c : cands) {
            if (track_used[c.track]) continue;
            if (cluster_used[c.cluster]) continue;
            const auto& cl = clusters[static_cast<size_t>(c.cluster)];
            double diff_chk = std::isfinite(gate_y[c.track]) ? std::abs(cl.mean_y - gate_y[c.track]) : std::abs(cl.mean_y - tracks[c.track].last_y);
            if (tracks[c.track].active && diff_chk > lane_disconnect_gap) continue;
            track_used[c.track] = true;
            cluster_used[c.cluster] = true;
            tracks[c.track].active = true;
            tracks[c.track].last_y = cl.mean_y;
            tracks[c.track].pts.insert(tracks[c.track].pts.end(), cl.points.begin(), cl.points.end());
        }

        // 남은 클러스터는 새 트랙으로(해당 측에서 빈 ID에 할당)
        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            if (cluster_used[ci]) continue;
            const auto& cl = clusters[ci];
            bool is_left = IsLeft(cl.mean_y);
            int id = is_left ? 0 : 1; // left lane1, right lane2
            if (!tracks[id].active && !track_used[id]) {
                track_used[id] = true;
                tracks[id].active = true;
                tracks[id].last_y = cl.mean_y;
                tracks[id].pts.insert(tracks[id].pts.end(), cl.points.begin(), cl.points.end());
            }
        }
    }

    // ----------------------------------------------------------------------------------
    // STEP 4. 차선 피팅
    // ----------------------------------------------------------------------------------

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

    std::fill(std::begin(has_prev_lane_id_), std::end(has_prev_lane_id_), false);
    for (int i = 0; i < 2; ++i) {
        interface::PolyfitLane lane;
        int lane_id = (i == 0) ? 1 : 2; // left=1, right=2
        if (fit_lane(tracks[i].pts, std::to_string(lane_id), lane)) {
            prev_lane_id_[i] = lane;
            has_prev_lane_id_[i] = true;
            poly_lanes_.polyfitlanes.push_back(lane);
        }
    }

    return poly_lanes_;
}

/** @brief X축 방향으로 slice 별로 points들을 나눠주는 함수
 *  @param lane_points
 *  @return 인덱스 번호와 lane_points로 이루어진 slices map */
std::map<int, std::vector<interface::Point2D>> PerceptionNode::SliceByX(const interface::Lane& lane_points){

    std::map<int, std::vector<interface::Point2D>> slices;

    // 1. 들어온 lane_points의 x 최대 최소값을 구한다
    min_x = lane_points.point.front().x;
    max_x = lane_points.point.front().x;
    for (const auto& pt : lane_points.point) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
    }

    // 2. 미리 설정한 슬라이스 폭으로 들어오는 lane points들을 슬라이스별로 나누어서 slice index 부여.
    for (const auto& pt : lane_points.point) {
        int slice_idx = static_cast<int>(std::floor((pt.x - min_x) / slice_width));   
        slices[slice_idx].push_back(pt);
    }
    return slices;
}

std::map<int, std::vector<PerceptionNode::Cluster>> PerceptionNode::FindCluster(std::map<int, std::vector<interface::Point2D>> slices){
    
    // 슬라이스 별로 차선이 저장된 cluster 들의 모임
    std::map<int, std::vector<PerceptionNode::Cluster>> clusters_by_slice;  

    if (hist_bin_width <= 0.0) {  // 히스토그램 bin의 너비가 0이면 이후 계산하지 않음
        RCLCPP_WARN(this->get_logger(), "[FindCluster] hist_bin_width <= 0.0: 클러스터링을 수행하지 않습니다.");
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
        std::vector<std::vector<size_t>> bins(bin_count);   // 각 빈에 속하는 점들의 인덱스 정수값을 저장한다.
        for (size_t i = 0; i < sorted_pts.size(); ++i) {
            int bin_idx = std::min(bin_count - 1, static_cast<int>(std::floor((sorted_pts[i].y - min_y) / hist_bin_width)));
            bins[bin_idx].push_back(i);
        }

        // 히스토그램을 훑으면서 빈 구간을 기준으로 클러스터 분리
        std::vector<Cluster> clusters;
        Cluster cur_cluster;
        double sum_y = 0.0, sum_x = 0.0;

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
            sum_y = sum_x = 0.0;
        };

        for (int bin_idx = 0; bin_idx < bin_count; ++bin_idx) {
            const auto& bin_points_idx = bins[bin_idx];

            if (!bin_points_idx.empty()) {
                for (size_t point_idx : bin_points_idx) {
                    add_point(sorted_pts[point_idx]);
                }
            } else {
                flush_cluster();  // 빈 bin 만나면 바로 이전 구간 확정
            }
        }
        flush_cluster();  // 마지막 구간 처리
        clusters_by_slice[idx] = clusters;
    }
    return clusters_by_slice;
}

interface::PolyfitLane PerceptionNode::FindDrivingWay(const interface::PolyfitLanes& poly_lanes) {
    
    interface::PolyfitLane driving_way_;
    driving_way_.frame_id = poly_lanes.frame_id;

    const interface::PolyfitLane* left_lane = nullptr;
    const interface::PolyfitLane* right_lane = nullptr;

    auto lane_idx = [](const interface::PolyfitLane& ln) -> int {
        if (ln.id.size() == 1 && std::isdigit(ln.id[0])) return ln.id[0] - '0';
        return -1;
    };

    for (const auto& lane : poly_lanes.polyfitlanes) {
        int idx = lane_idx(lane);
        if (idx == 1) { // 좌 내측 우선
            left_lane = &lane;
        } else if (idx == 0 && left_lane == nullptr) { // 좌 외측
            left_lane = &lane;
        } else if (idx == 2) { // 우 내측 우선
            right_lane = &lane;
        } else if (idx == 3 && right_lane == nullptr) { // 우 외측
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
        int idx = lane_idx(*single_lane);
        if (idx == 0 || idx == 1) {
            offset_sign = -offset;
        } else {
            offset_sign = offset;
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
