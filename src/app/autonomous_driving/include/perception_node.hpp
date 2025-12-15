#ifndef __PERCEPTION_NODE_HPP__
#define __PERCEPTION_NODE_HPP__
#pragma once

// STD Header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense> 

// Interface Header (ROS 독립적)
#include "interface_lane.hpp"
#include "interface_vehicle.hpp" 

// Bridge Header
#include "ros2_bridge_vehicle.hpp"
#include "ros2_bridge_lane.hpp"
#include "ros2_bridge_mission.hpp"

// Parameter Header
#include "autonomous_driving_config.hpp"

class PerceptionNode : public rclcpp::Node {
    public:
        explicit PerceptionNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
        virtual ~PerceptionNode();

        void ProcessParams();
        void Run();
        
        struct Cluster {
            double mean_y{0.0};
            double mean_x{0.0};
            std::vector<interface::Point2D> points;
        };

    private:

        //-- Functions ------------------------------------------------//
        void Init(const rclcpp::Time &current_time);  // ← 이 줄 추가

        // Callback functions

        inline void CallbackVehicleState(const ad_msgs::msg::VehicleState::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
            i_vehicle_state_ = ros2_bridge::GetVehicleState(*msg);
            b_is_simulator_on_ = true;
        }
        inline void CallbackLanePoints(const ad_msgs::msg::LanePointData::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_lane_points_);
            i_lane_points_ = ros2_bridge::GetLanePoints(*msg);
            b_is_lane_points_ = true;
        }

        // algorithm
        interface::PolyfitLanes FindLanes(const interface::Lane& lane_points);
        std::map<int, std::vector<interface::Point2D>> SliceByX(const interface::Lane& lane_points);
        double SliceCenter(int idx) {
            return min_x + (static_cast<double>(idx) + 0.5) * slice_width;
        }
        double EvalLane(const interface::PolyfitLane& lane, double x) { 
            return lane.a0 + lane.a1 * x + lane.a2 * x * x + lane.a3 * x * x * x;
        }
        std::map<int, std::vector<PerceptionNode::Cluster>> FindCluster(std::map<int, std::vector<interface::Point2D>> slices);
        interface::PolyfitLane FindDrivingWay(const interface::PolyfitLanes& poly_lanes);
        //-- Variable ------------------------------------------------//

        // Subscriber 
        rclcpp::Subscription<ad_msgs::msg::VehicleState>::SharedPtr s_vehicle_state_;
        rclcpp::Subscription<ad_msgs::msg::LanePointData>::SharedPtr s_lane_points_;

        // Input
        interface::VehicleState i_vehicle_state_;
        interface::Lane i_lane_points_;

        // Mutex
        std::mutex mutex_vehicle_state_;
        std::mutex mutex_lane_points_;

        double min_x;
        double max_x;
        double slice_width = 0.5;                         // x 슬라이스 폭 [m]
        double cluster_threshold = 0.5;                   // 슬라이스 내 y 클러스터 간격 [m]
        double gate_width = 0.5;                          // 이전 프레임 기반 게이팅 폭 [m]
        double hist_bin_width = cluster_threshold * 0.25; // 빈 히스토그램 폭을 세밀하게 분리
        double side_lane_window = 0.5;                    // 동일 측면에서 다른 차선으로 점프하지 않도록 허용하는 y 거리 [m]
        double lane_disconnect_gap = 0.6;                 // 차선 포인트가 끊길 때 옆 차선으로 점프하지 않도록 허용하는 최대 y 이격
        double fit_near_sigma = 5.0;                      // 근거리 가중치 감쇠율 [m]
        double coeff_smooth_alpha = 0.3;                  // 계수 스무딩 계수 (현재/이전 가중 평균)
        int min_fit_points = 8;                           // 차선 피팅에 필요한 최소 포인트 수

        //-- Output  ----------------------------------------------------//

        // Publisher
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneDataArray>::SharedPtr p_poly_lanes_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr p_driving_way_;

        // Previous frame lanes for gating
        bool has_prev_left_lane_{false};            // 이전 프레임에서 왼쪽 차선이 유효하게 추정되었는지
        bool has_prev_right_lane_{false};           // 이전 프레임에서 오른쪽 차선이 유효하게 추정되었는지 
        interface::PolyfitLane prev_left_lane_;     // 현재 프레임의 왼쪽 차선을 찾을 때 이전 프레임의 결과
        interface::PolyfitLane prev_right_lane_;    // 현재 프레임의 오른쪽 차선을 찾을
        bool has_prev_lane_id_[2]{false, false}; // lane id 1,2만 사용 (좌/우)
        interface::PolyfitLane prev_lane_id_[2];

        // Previous driving way for smoothing
        bool has_prev_driving_way_{false};
        interface::PolyfitLane prev_driving_way_;
        double driving_way_smooth_alpha_{0.3}; // exp smoothing gain

        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;

        // Util and Configuration
        AutonomousDrivingConfig cfg_;

        // Flag
        bool b_is_simulator_on_ = false;
        bool b_is_lane_points_ = false;
  
    };

#endif // __PERCEPTION_NODE_HPP__
