/*
* planning_node.hpp
*/
#ifndef __PLANNING_NODE_HPP__
#define __PLANNING_NODE_HPP__
#pragma once

// STD Header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Dense> 

// Interface Header (ROS 독립적)
#include "interface_lane.hpp"
#include "interface_vehicle.hpp" 

// Bridge Header
#include "ros2_bridge_vehicle.hpp"
#include "ros2_bridge_lane.hpp"
#include "ros2_bridge_mission.hpp"

#include <std_msgs/msg/float32.hpp>

// Parameter Header
#include "autonomous_driving_config.hpp"

class PlanningNode : public rclcpp::Node {
    public:
        explicit PlanningNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
        virtual ~PlanningNode();

        void ProcessParams();
        void Run();

    private:
        //----------------------------------------------------//
        // Functions
        void Init(const rclcpp::Time &current_time);  // ← 이 줄 추가

        // Callback functions
        inline void CallbackManualInput(const ad_msgs::msg::VehicleCommand::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_manual_input_);
            if (cfg_.use_manual_inputs == true) {
                i_manual_input_ = ros2_bridge::GetVehicleCommand(*msg);
                b_is_manual_input_ = true;
            }
        }
        inline void CallbackVehicleState(const ad_msgs::msg::VehicleState::SharedPtr msg) 
        {            
            std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
            i_vehicle_state_ = ros2_bridge::GetVehicleState(*msg);
            b_is_simulator_on_ = true;
        }
        //[다훈 수정0] i_limit_speed_ mutex 보호 추가
        //inline void CallbackLimitSpeed(const std_msgs::msg::Float32::SharedPtr msg) 
        //{            
        //    std::lock_guard<std::mutex> lock(mutex_limit_speed_);
        //    i_limit_speed_ = msg->data;
        //}
        inline void CallbackLanePoints(const ad_msgs::msg::LanePointData::SharedPtr msg) 
        {            
            std::lock_guard<std::mutex> lock(mutex_lane_points_);
            i_lane_points_ = ros2_bridge::GetLanePoints(*msg);
            b_is_lane_points_ = true;
        }
        inline void CallbackMission(const ad_msgs::msg::Mission::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_mission_);
            i_mission_ = ros2_bridge::GetMission(*msg);
            b_is_mission_ = true;
        }
        //[11.28 다훈 수정] driving_way_raw callback 추가
        inline void CallbackPolyfitLaneData(const ad_msgs::msg::PolyfitLaneData::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_driving_way_raw_);
            i_driving_way_raw_ = ros2_bridge::GetPolyfitLaneData(*msg);
            b_is_driving_way_raw_ = true;
        }

        ////////////////////// TODO //////////////////////
        // TODO: Add more functions
    
        //[다훈 수정*] VelocityPlanning 함수 추가
        // - algorithm::FindDrivingWay()
        double VelocityPlanning(const interface::VehicleState &vehicle_state, const interface::Lane &lane_points, const interface::Mission &mission, const interface::PolyfitLane &driving_way_real);
        //////////////////////////////////////////////////

        //----------------------------------------------------//
        // Variable

        //============================
        // Subscriber 
        //============================
        rclcpp::Subscription<ad_msgs::msg::VehicleCommand>::SharedPtr s_manual_input_;
        rclcpp::Subscription<ad_msgs::msg::VehicleState>::SharedPtr s_vehicle_state_;
        //[다훈 수정1] i_limit_speed_ 추가
        //rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr s_limit_speed_;
        rclcpp::Subscription<ad_msgs::msg::LanePointData>::SharedPtr s_lane_points_;
        rclcpp::Subscription<ad_msgs::msg::Mission>::SharedPtr s_mission_;
        //driving_way 받아오기 
        rclcpp::Subscription<ad_msgs::msg::PolyfitLaneData>::SharedPtr s_driving_way_raw_;

        //============================
        // Input
        //============================
        interface::VehicleCommand i_manual_input_;
        interface::VehicleState i_vehicle_state_;
        //[다훈 수정2] i_limit_speed_ 추가
        //double i_limit_speed_ = 0.0;
        interface::Lane i_lane_points_;
        interface::Mission i_mission_;
        interface::PolyfitLane i_driving_way_raw_;

        //============================
        // Mutex
        //============================
        std::mutex mutex_manual_input_;
        std::mutex mutex_vehicle_state_;
        //[다훈 수정3] i_limit_speed_ mutex 추가
        //std::mutex mutex_limit_speed_;
        std::mutex mutex_lane_points_;
        std::mutex mutex_mission_;
        std::mutex mutex_driving_way_raw_;

        //===============================================
        // Publisher
        //===============================================
        rclcpp::Publisher<ad_msgs::msg::VehicleCommand>::SharedPtr          p_vehicle_command_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr         p_driving_way_real_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr               p_reference_speed_; 

        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;

        // Util and Configuration
        AutonomousDrivingConfig cfg_;
                
        // Flag
        bool b_is_manual_input_ = false;
        bool b_is_simulator_on_ = false;
        bool b_is_lane_points_ = false;
        bool b_is_mission_ = false;
        bool b_is_driving_way_raw_ = false;
};

#endif // PLANNING_NODE_HPP__