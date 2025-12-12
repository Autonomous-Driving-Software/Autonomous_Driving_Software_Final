
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
        interface::PolyfitLanes FindLanes(const interface::VehicleState &vehicle_state, const interface::Lane& lane_points);
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

        //-- Output  ----------------------------------------------------//

        // Publisher
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneDataArray>::SharedPtr p_poly_lanes_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr p_driving_way_;

        // Previous frame lanes for gating
        bool has_prev_left_lane_{false};
        bool has_prev_right_lane_{false};
        interface::PolyfitLane prev_left_lane_;
        interface::PolyfitLane prev_right_lane_;

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