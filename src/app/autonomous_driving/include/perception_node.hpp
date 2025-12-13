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
        std::map<int, std::vector<PerceptionNode::Cluster>> ClusterLanePoints(std::map<int, std::vector<interface::Point2D>> slices);
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
        const double slice_width = 0.2;                         // x 슬라이스 폭 [m]
        const double cluster_threshold = 0.2;                   // 슬라이스 내 y 클러스터 간격 [m]
        const double gate_width = 0.2;                          // 이전 프레임 기반 게이팅 폭 [m]
        const double hist_bin_width = cluster_threshold * 0.25; // 빈 히스토그램 폭을 세밀하게 분리
        const int empty_bin_gap = 1;                            // 연속 빈 bin 허용 개수
        const int start_search_span = 5;                        // 슬라이스 범위 내에서 최초 씨드를 찾기 위한 탐색 범위 (앞/뒤 N슬라이스)

        //-- Output  ----------------------------------------------------//

        // Publisher
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneDataArray>::SharedPtr p_poly_lanes_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr p_driving_way_;

        // Previous frame lanes for gating
        bool has_prev_left_lane_{false};            // 이전 프레임에서 왼쪽 차선이 유효하게 추정되었는지
        bool has_prev_right_lane_{false};           // 이전 프레임에서 오른쪽 차선이 유효하게 추정되었는지 
        interface::PolyfitLane prev_left_lane_;     // 현재 프레임의 왼쪽 차선을 찾을 때 이전 프레임의 결과
        interface::PolyfitLane prev_right_lane_;    // 현재 프레임의 오른쪽 차선을 찾을 때 이전 프레임의 결과

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