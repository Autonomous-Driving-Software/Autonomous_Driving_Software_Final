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

// Frenet Converter Header
#include "frenet_converter.hpp" 

// Bridge Header
#include "ros2_bridge_vehicle.hpp"
#include "ros2_bridge_lane.hpp"
#include "ros2_bridge_mission.hpp"

#include <std_msgs/msg/float32.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Parameter Header
#include "autonomous_driving_config.hpp"

class PlanningNode : public rclcpp::Node {
    public:
        //================================
        // 주행 모드 정의
        //================================
        enum class DrivingMode {
            NORMAL_DRIVING, // 일반 주행
            SCC, // Safe Cruise Control
            LANE_CHANGE // 차선 변경
        };

        //================================
        // Behavior 정보를 담는 context 구조체(struct) 정의
        // - BehaviorPlanning: 모드 결정 + 객체 정보만 저장
        // - VelocityPlanning: TTC/속도 계산 담당
        //================================
        struct BehaviorContext {
            DrivingMode current_mode = DrivingMode::NORMAL_DRIVING;
            
            // ============================================
            // SCC 관련 변수 [for Dynamic Obstacle]
            // - BehaviorPlanning: 앞차 정보만 저장
            // - VelocityPlanning: TTC/v_lead 계산
            // ============================================
            bool has_dynamic_object = false;
            double lead_s = 1e6;           // 앞차까지 Frenet s 거리 [m]
            double lead_velocity = 0.0;    // 앞차 속도 [m/s]

            // ============================================
            // Lane Change 관련 변수 [for Static Obstacle]
            // ============================================
            bool has_static_object = false;
            double dist_static = 1e6;
            double static_object_x_rel = 0.0; //장애물 ego 좌표계 x
            double static_object_y_rel = 0.0; //장애물 ego 좌표계 y

            // Frenet 좌표계 사용 
            double static_object_s = 0.0; //장애물의 frenet s 좌표
            double static_object_d = 0.0; //장애물의 frenet d

            //Lane ID 추적 (Lane 0=오른쪽, Lane 1=가운데, Lane 2=왼쪽)
            int current_lane_id = 1; //기본값: 가운데 차선 
            bool is_lane_changing = false; //차선 변경 중인지 여부

            //다른 차선 장애물 정보 
            bool left_lane_blocked = false;   // 왼쪽 차선(Lane 2)에 장애물 있는지
            bool right_lane_blocked = false;  // 오른쪽 차선(Lane 0)에 장애물 있는지

            interface::Mission mission;  // 전체 mission 저장
        };

        //================================
        // Waypoint
        //================================
        struct Waypoint {
            double x;
            double y;
            double s; // 누적 거리
        };
        //================================
        //frenet 좌표계
        //================================
        struct FrenetCoordinate {
            double s; // 누적 거리
            double d; // lateral offset
        }; 

        explicit PlanningNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
        virtual ~PlanningNode();

        void ProcessParams();
        void Run();

    private:
        
        //----------------------------------------------------//
        // Functions
        void Init(const rclcpp::Time &current_time);  //이 줄 추가

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
        //[11.28 다훈 수정] driving_way callback 추가
        inline void CallbackPolyfitLaneData(const ad_msgs::msg::PolyfitLaneData::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_driving_way_);
            i_driving_way_ = ros2_bridge::GetPolyfitLaneData(*msg);
            b_is_driving_way_ = true;
        }

        //================================================
        // 함수 선언
        //================================================
        // 1. [11.28 다훈 수정] Behavior(모드) 판단 함수 추가
        BehaviorContext BehaviorPlanning(const interface::VehicleState &vehicle_state, const interface::Mission &mission, const interface::PolyfitLane &driving_way);

        // 2. [11.28 다훈 수정*] VelocityPlanning 함수 추가 + BehaviorContext 추가
        double VelocityPlanning(const interface::VehicleState &vehicle_state, const interface::Lane &lane_points, const interface::Mission &mission, const interface::PolyfitLane &driving_way_real, const BehaviorContext &ctx);

        // 3. [12.10 다훈 수정] Forward-Backward Speed Profile Smoothing 함수
        double SmoothSpeedProfile(const interface::PolyfitLane &driving_way, double v_ref, double current_velocity);

        // 4. [12.07 다훈 수정] Frenet 좌표계 변환 함수 
        FrenetCoordinate CartesianToFrenet(double x_rel, double y_rel, const interface::PolyfitLane &driving_way);

        std::pair<double, double> FrenetToCartesian(double s, double d, const interface::PolyfitLane &driving_way);

        // 4. [12.08 다훈 수정] FrenetConverter 업데이트 함수
        void UpdateFrenetConverter(const interface::PolyfitLane &driving_way);

        interface::PolyfitLane LaneChange(const interface::VehicleState &vehicle_state, const interface::PolyfitLane &driving_way, const BehaviorContext &ctx);


        //[11.28 다훈 수정] GlobalToLocal 함수 추가(dynamic / static obstacle 좌표 변환용)
        // - algorithm::GlobalToLocal()
        std::pair<double, double> GlobalToLocal(const interface::VehicleState &vehicle_state, double obj_x_global, double obj_y_global);

        //[12.04 다훈 수정] LocalToGlobal 함수 추가 (차선 변경 시 목표 지점 좌표 변환용)
        std::pair<double, double> LocalToGlobal(const interface::VehicleState &vehicle_state, double x_rel, double y_rel);
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
        rclcpp::Subscription<ad_msgs::msg::PolyfitLaneData>::SharedPtr s_driving_way_;

        //============================
        // Input
        //============================
        interface::VehicleCommand i_manual_input_;
        interface::VehicleState i_vehicle_state_;
        //[다훈 수정2] i_limit_speed_ 추가
        //double i_limit_speed_ = 0.0;
        interface::Lane i_lane_points_;
        interface::Mission i_mission_;
        interface::PolyfitLane i_driving_way_;

        //============================
        // Mutex
        //============================
        std::mutex mutex_manual_input_;
        std::mutex mutex_vehicle_state_;
        //[다훈 수정3] i_limit_speed_ mutex 추가
        //std::mutex mutex_limit_speed_;
        std::mutex mutex_lane_points_;
        std::mutex mutex_mission_;
        std::mutex mutex_driving_way_;

        //===============================================
        // Publisher
        //===============================================
        rclcpp::Publisher<ad_msgs::msg::VehicleCommand>::SharedPtr          p_vehicle_command_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr         p_driving_way_real_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr               p_reference_speed_;
        
        // Visualization Publishers
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr       p_lane_change_target_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  p_lane_change_path_; 
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr       p_static_object_marker_;  // Static object 위치 시각화
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  p_frenet_debug_marker_;   // Frenet 디버깅용

        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;

        // Util and Configuration
        AutonomousDrivingConfig cfg_;
        
        //==============================================
        // 차선 변경 관련 변수
        //==============================================
        // Lane ID (Lane 0=오른쪽, Lane 1=가운데, Lane 2=왼쪽)
        int current_lane_id_ = 1;    // 현재 차선 ID
        int target_lane_id_ = 1;     // 목표 차선 ID
        double target_d_final_ = 0.0; // 차선 변경 목표 lateral offset (±4m)
        
        // 차선 변경 진행 관리
        int lane_change_counter_ = 0;      // 차선 변경 진행 카운터 (타임아웃용)
        bool is_lane_changing_ = false;    // 차선 변경 중 여부
        bool lane_change_path_saved_ = false; // 경로 저장 완료 여부
        
        // 빨간 구체 (목표 지점) Global 좌표
        double target_point_global_x_ = 0.0;
        double target_point_global_y_ = 0.0;
        
        // Frenet 경로 생성용 sf 저장
        double lane_change_sf_ = 0.0;  // 차선 변경 완료 s 거리
        
        // 차선 변경 경로 Global 좌표 저장
        std::vector<std::pair<double, double>> lane_change_path_global_;
        
        //==============================================
        // FrenetConverter 관련 변수
        //==============================================
        FrenetConverter frenet_converter_;           // Frenet 좌표 변환기
        interface::PolyfitLane cached_driving_way_;  // 캐시된 driving_way (변경 감지용)
        bool frenet_converter_initialized_ = false;  // FrenetConverter 초기화 여부
        
        //==============================================
        // Flag (데이터 수신 확인용)
        //==============================================
        bool b_is_manual_input_ = false;
        bool b_is_simulator_on_ = false;
        bool b_is_lane_points_ = false;
        bool b_is_mission_ = false;
        bool b_is_driving_way_ = false;
};

#endif // PLANNING_NODE_HPP__