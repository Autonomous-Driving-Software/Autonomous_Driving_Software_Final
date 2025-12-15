/*
 * planning_node.cpp
 */
#include "autonomous_driving_config.hpp"
#include "planning_node.hpp"
#include "frenet_converter.hpp"

using namespace std;

PlanningNode::PlanningNode(const std::string &node_name, const rclcpp::NodeOptions &options): Node(node_name, options) {
    RCLCPP_WARN(this->get_logger(), "Initialize node...");

    // QoS init
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

    //get parameters(파라미터 값 가져오기)
    // Vehicle
    //this->get_parameter("autonomous_driving/wheel_base", cfg_.param_wheel_base);
    //this->get_parameter("autonomous_driving/max_lateral_accel", cfg_.max_lateral_accel);

    /* 노드를 ROS 네트워크에 연결하는 부분 
    공통적으로 하는 일 create_subscription<메시지타입>(토픽이름, QoS, callback)*/
    //============Subscriber init===============
    //(1)s_manual_input_
    s_manual_input_ = this->create_subscription<ad_msgs::msg::VehicleCommand>(
        "/manual_input", qos_profile, std::bind(&PlanningNode::CallbackManualInput, this, std::placeholders::_1));

    //(2) s_vehicle_state_
    s_vehicle_state_ = 
    this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&
        PlanningNode::CallbackVehicleState, this, 
    std::placeholders::_1));

    //[다훈 수정0] /limit_speed_ subscriber 추가 
    //(3) s_limit_speed_
    //s_limit_speed_ = this->create_subscription<std_msgs::msg::Float32>(
    //    "/limit_speed", qos_profile, std::bind(&PlanningNode::CallbackLimitSpeed, this, std::placeholders::_1));

    //(4) s_lane_points_
    s_lane_points_ = 
    this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&PlanningNode::CallbackLanePoints, this, std::placeholders::_1));
        
    //(5) s_mission_
    s_mission_ = this->create_subscription<ad_msgs::msg::Mission>(
        "mission", qos_profile, std::bind(&PlanningNode::CallbackMission, this, std::placeholders::_1));

    //[11.28 다훈 수정] driving_way subscriber 추가
    s_driving_way_ = this->create_subscription<ad_msgs::msg::PolyfitLaneData>(
        "driving_way", qos_profile, std::bind(&PlanningNode::CallbackPolyfitLaneData, this, std::placeholders::_1));

    //======================================================
    //publisher init
    //======================================================
    p_vehicle_command_ = this->create_publisher<ad_msgs::msg::VehicleCommand>(
        "vehicle_command", qos_profile);
    p_driving_way_real_ = this->create_publisher<ad_msgs::msg::PolyfitLaneData>(
        "driving_way_real", qos_profile);

    //[다훈 추가]p_reference_speed_ (이게 맞나?)(lon에 넘기려고)
    p_reference_speed_ = this->create_publisher<std_msgs::msg::Float32>(
        "reference_speed", qos_profile);
    
    // Visualization Publishers
    p_lane_change_target_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "lane_change_target", qos_profile);
    p_lane_change_path_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "lane_change_path", qos_profile);

    // Initialize
    Init(this->now());

    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); }); 

}
PlanningNode::~PlanningNode() {}

void PlanningNode::Init(const rclcpp::Time &current_time) {
}

void PlanningNode::ProcessParams() {
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("autonomous_driving/use_manual_inputs", cfg_.use_manual_inputs);
}

void PlanningNode::Run() {
    auto current_time = this->now();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "Running ..."); //로그는 최소 1000ms에 한번만 출력
    ProcessParams();

    //===================================================
    // 아직 필요한 input 토픽 안들어왔으면 알고리즘 실행하지 않고 기다림
    // Get subscribe variables
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
    if (b_is_mission_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Mission ...");
        return;
    }
    //[11.28 다훈 수정] driving_way 추가
    if (b_is_driving_way_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Driving Way Raw ...");
        return;
    }

    //===================================================
    // 1. <Get subscribe variables>
    // [일종의 input데이터 수집 단계 (멤버 변수 -> 지역변수로 복사 (mutex로 보호))]
    //===================================================
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
    interface::Mission mission; {
        std::lock_guard<std::mutex> lock(mutex_mission_);
        mission = i_mission_;
    }
    interface::PolyfitLane driving_way; {
    std::lock_guard<std::mutex> lock(mutex_driving_way_);
    driving_way = i_driving_way_;
    }   

    //===================================================
    // 2.[12.08 다훈 추가] <FrenetConverter 업데이트>
    // driving_way가 변경되면 FrenetConverter도 다시 초기화
    //===================================================
    UpdateFrenetConverter(driving_way);

    //===================================================
    // 3.[11.28 다훈 수정] <Behavior Planning>
    //===================================================
    // ctx
    PlanningNode::BehaviorContext ctx = BehaviorPlanning(vehicle_state, mission, driving_way);

    // 모드 확인 로그
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, 
        "[Run] Current Mode: %d, has_static: %d, is_changing: %d", 
        static_cast<int>(ctx.current_mode), ctx.has_static_object, is_lane_changing_);

    //Lane Change 모드일 때 새로운 경로 생성 (Control로 넘어가는건 driving_way_real)
    interface::PolyfitLane driving_way_real;
    
    //=================================================================
    // 4.[12.11 다훈 수정] <Lane Change 시작 조건>
    // - Static object 감지되면 바로 시작
    // - VelocityPlanning에서 감속 처리
    // - 차선 변경 중에도 새로운 장애물 발견 시 경로 재계획
    //=================================================================
    if (is_lane_changing_ || (ctx.current_mode == DrivingMode::LANE_CHANGE && ctx.has_static_object)) {
        // 차선 변경 중 새로운 장애물 발견 시 경로 재계획
        bool need_replan = false;
        if (is_lane_changing_ && ctx.has_static_object) {
            // 현재 목표 차선 방향 확인
            int lane_direction = target_lane_id_ - current_lane_id_;  // +1=왼쪽, -1=오른쪽
            
            // 목표 차선 쪽에 새로운 장애물이 있는지 확인
            if ((lane_direction > 0 && ctx.left_lane_blocked) ||   // 왼쪽으로 가는데 왼쪽 막힘
                (lane_direction < 0 && ctx.right_lane_blocked)) {  // 오른쪽으로 가는데 오른쪽 막힘
                need_replan = true;
            }
        }
        
        // 경로 재계획이 필요하면 기존 경로 초기화
        if (need_replan) {
            lane_change_path_saved_ = false;
            lane_change_path_global_.clear();
            // current_lane_id_는 유지 (아직 차선 변경 완료 안됨)
        }

        driving_way_real = LaneChange(vehicle_state, driving_way, ctx);
        
        // 디버깅: 차선 변경 경로 계수 출력
        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000,
            "[Run] driving_way_REAL: a0=%.3f, a1=%.3f, a2=%.6f, a3=%.9f",
            driving_way_real.a0, driving_way_real.a1, driving_way_real.a2, driving_way_real.a3);
    } else {
        driving_way_real = driving_way;
        
        // 디버깅: 원래 경로 계수 출력
        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000,
            "[Run] driving_way: a0=%.3f, a1=%.3f, a2=%.6f, a3=%.9f",
            driving_way.a0, driving_way.a1, driving_way.a2, driving_way.a3);
    }

    //[다훈 수정2] driving_way_real 사용 (차선 변경 경로의 곡률 반영)
    //(3)Add velocity planning algorithm
    double reference_speed = VelocityPlanning(vehicle_state, lane_points, mission, driving_way_real, ctx);

    //===================================================
    // Publish output
    //===================================================
    //Publish driving way
    p_driving_way_real_->publish(ros2_bridge::UpdatePolyfitLane(driving_way_real));

    //[다훈 수정] reference_speed publish
    std_msgs::msg::Float32 ref_msg;
    ref_msg.data = reference_speed;
    p_reference_speed_->publish(ref_msg);
}



//============================================
//#BehaviorPlanning 함수 구현
// input: vehicle_state / mission / driving_way
// output: BehaviorContext (현재 모드 정보) / SCC일 때 TTC에 맞는 v_lead / Lane Change일 때 driving_way_real
//============================================
PlanningNode::BehaviorContext PlanningNode::BehaviorPlanning(const interface::VehicleState &vehicle_state, const interface::Mission &mission, const interface::PolyfitLane &driving_way) {
    // ctx initialization
    BehaviorContext ctx;
    ctx.current_mode = DrivingMode::NORMAL_DRIVING; // 기본 모드 설정
    ctx.mission = mission; // mission 정보 저장
    
    // ✅ 수정: 클래스 멤버 변수에서 항상 복사 (Static 객체와 무관하게)
    ctx.current_lane_id = current_lane_id_;
    ctx.is_lane_changing = is_lane_changing_;

    // 모든 object 돌면서 확인 
    for (const auto &obj : mission.objects) {
        //1) Global->Ego Coordinate 변환
        auto [x_rel, y_rel] = GlobalToLocal(vehicle_state, obj.x, obj.y);

        //==================================================================
        //## 같은 차선에 있는 장애물 인지 판단 Logic 
        //==================================================================
        //2) ego 앞쪽(+x)만 고려
        if (x_rel <= 0.0) {
            continue; // 뒤쪽 객체 무시
        }
        
        //Frenet 좌표로 변환(모든 object에 대해)
        FrenetCoordinate frenet = CartesianToFrenet(x_rel, y_rel, driving_way);

        //차선 판별(d좌표 기준) - Lane 0=오른쪽, Lane 1=가운데, Lane 2=왼쪽
        const double LANE_WIDTH = 4.0; //차선 폭 4m

        //## 같은 차선 (d ≈ 0)
        if (std::abs(frenet.d) < 2.0) { 
            //==================================================================
            //### SCC (Dynamic object) - 앞차 정보만 저장
            // TTC/v_lead 계산은 VelocityPlanning에서 수행
            //==================================================================
            if (obj.object_type == "Dynamic") {
                ctx.has_dynamic_object = true;
                
                // 가장 가까운 앞차 정보만 저장
                if (frenet.s < ctx.lead_s) {
                    ctx.lead_s = frenet.s;           // 앞차까지 Frenet s 거리
                    ctx.lead_velocity = obj.velocity; // 앞차 속도
                }
            }
            //==================================================================
            //### Lane Change (Static object)
            // 장애물 2개 정보를 하나로 인식하는 이슈로 인해 가까운 장애물(s < 15m)만 차선 변경 대상으로 고려
            //==================================================================
            else if (obj.object_type == "Static") {
                //double v_ref = mission.speed_limit*0.3; // mission에서 직접 가져오기 (이게 v_lead)
                const double LANE_CHANGE_DETECTION_RANGE = 17.0;  // 차선 변경 감지 범위
                
                // 가까운 장애물만 차선 변경 대상으로 고려
                if (frenet.s < LANE_CHANGE_DETECTION_RANGE) {
                    ctx.has_static_object = true; // Static Object (장애물) 존재
                    ctx.dist_static = std::min(ctx.dist_static, x_rel);

                    //장애물의 ego 좌표 저장 
                    ctx.static_object_x_rel = x_rel;
                    ctx.static_object_y_rel = y_rel;
                    ctx.static_object_s = frenet.s;
                    ctx.static_object_d = frenet.d;
                }
            }
        }
        //## 왼쪽 차선 (d ≈ +4, Lane 2)
        else if (std::abs(frenet.d - LANE_WIDTH) < 2.0) {
            ctx.left_lane_blocked = true;
        }
        //## 오른쪽 차선 (d ≈ -4, Lane 0)
        else if (std::abs(frenet.d + LANE_WIDTH) < 2.0) {
            ctx.right_lane_blocked = true;
        }
    }
    
    //===================================
    //## Mode Decision (1순위 static, 2순위 dynamic, 3순위 일반주행)
    // - BehaviorPlanning: 모드 결정만
    // - VelocityPlanning: TTC/속도 계산
    //===================================
    //모드 결정 - Static 객체가 같은 차선에 있으면 바로 차선 변경
    if (ctx.has_static_object && !ctx.is_lane_changing) {
        ctx.current_mode = DrivingMode::LANE_CHANGE;
    }
    else if (ctx.has_dynamic_object) {
        ctx.current_mode = DrivingMode::SCC;
    }
    else {
        ctx.current_mode = DrivingMode::NORMAL_DRIVING;
    }

    return ctx;
}

//============================================
// SmoothSpeedProfile 함수 구현
// Forward-Backward Speed Profile Smoothing
// 
// PPT 공식:
// - Forward:  v_new = sqrt(2 * a_max * Δs + v0²)  (가속 제한)
// - Backward: v0_new = sqrt(v² - 2 * a_min * Δs)  (감속 제한)
//
// input: driving_way (경로), v_ref (목표 속도), current_velocity (현재 속도)
// output: smoothed reference_speed at s=0 (현재 위치에서의 목표 속도)
//============================================
double PlanningNode::SmoothSpeedProfile(const interface::PolyfitLane &driving_way, double v_ref, double current_velocity) {
    // 파라미터
    const double a_max = cfg_.param_a_max;   // 최대 가속도 [m/s²] (양수: 2.0)
    const double a_min = cfg_.param_a_min;   // 최대 감속도 [m/s²] (음수: -3.0)
    const double ds = 0.5;                    // 샘플링 간격 [m]
    const double s_horizon = 50.0;            // 경로 horizon [m]
    
    // 경로 샘플링 (s = 0 ~ s_horizon)
    int n_points = static_cast<int>(s_horizon / ds) + 1;
    std::vector<double> v_arr(n_points);
    
    // 1) 각 s 위치에서 곡률 기반 속도 제한 계산
    for (int i = 0; i < n_points; i++) {
        double x = i * ds;  // x ≈ s (직진 근사)
        
        // 곡률 계산: kappa = |y''| / (1 + y'^2)^(3/2)
        double y_prime = 3.0 * driving_way.a3 * x * x + 2.0 * driving_way.a2 * x + driving_way.a1;
        double y_double_prime = 6.0 * driving_way.a3 * x + 2.0 * driving_way.a2;
        double denom = std::pow(1.0 + y_prime * y_prime, 1.5);
        double kappa = (denom > 1e-6) ? std::abs(y_double_prime) / denom : 0.0;
        
        // 곡률 기반 속도 제한: v_kappa = sqrt(a_lat_max / |kappa|)
        double v_kappa = v_ref;
        if (kappa > 1e-6) {
            v_kappa = std::sqrt(cfg_.param_max_lateral_accel / kappa);
            v_kappa = std::min(v_kappa, v_ref);
        }
        
        // 초기 속도 프로파일: v_ref와 v_kappa 중 작은 값
        v_arr[i] = std::min(v_ref, v_kappa);
    }
    
    // 2) Forward Pass: 가속 제한 (a_max)
    // 중요: Forward는 "현재 속도에서 얼마나 가속할 수 있는지" 계산
    // 하지만 목표는 v_ref까지 가는 것이므로, Forward Pass는 v_arr[0]부터 시작하지 않고
    // 각 점에서 이전 점으로부터 도달 가능한 최대 속도를 계산
    
    // Forward Pass: 첫 점은 현재 속도로 제한하지 않음 (목표 속도 프로파일 계산)
    for (int i = 1; i < n_points; i++) {
        double v0 = v_arr[i - 1];
        double v_target = v_arr[i];
        
        // 이전 점에서 현재 점으로 가속할 때 a_max 제한
        double v_max_from_prev_sq = v0 * v0 + 2.0 * a_max * ds;
        double v_max_from_prev = (v_max_from_prev_sq > 0) ? std::sqrt(v_max_from_prev_sq) : 0.0;
        
        // Forward limit 적용
        v_arr[i] = std::min(v_target, v_max_from_prev);
    }
    
    // 3) Backward Pass: 감속 제한 (a_min, 음수)
    // 뒤에서부터 역으로 계산: 다음 점에 도달하기 위해 현재 점에서 허용되는 최대 속도
    for (int i = n_points - 2; i >= 0; i--) {
        double v_next = v_arr[i + 1];  // 다음 점 속도
        
        // 현재 점에서 다음 점까지 감속할 때: a = (v_next² - v_current²) / (2*ds)
        // a >= a_min 이어야 함 (a_min은 음수)
        // v_current² <= v_next² - 2 * a_min * ds
        double v_max_sq = v_next * v_next - 2.0 * a_min * ds;  // a_min이 음수이므로 +가 됨
        double v_max_from_next = (v_max_sq > 0) ? std::sqrt(v_max_sq) : 0.0;
        
        // Backward limit 적용
        v_arr[i] = std::min(v_arr[i], v_max_from_next);
    }
    
    // 4) 현재 속도 기준으로 최종 목표 속도 결정
    // v_arr[0]은 경로 시작점(현재 위치)에서의 목표 속도
    double smoothed_speed = v_arr[0];
    
    // 현재 속도에서 목표 속도까지 가속 제한 확인
    // 만약 현재 속도가 너무 낮으면 a_max로 가속 가능한 범위 내에서 목표 설정
    double v_max_accel = std::sqrt(current_velocity * current_velocity + 2.0 * a_max * ds);
    smoothed_speed = std::min(smoothed_speed, v_max_accel);
    
    // 최소 속도 보장 (정지 상태에서 시작할 때)
    if (current_velocity < 0.1 && smoothed_speed < 0.5) {
        smoothed_speed = std::min(v_ref, 5.0);  // 정지 상태면 5 m/s 또는 v_ref로 목표 설정
    }
    
    return smoothed_speed;
}

//============================================
// VelocityPlanning 
// input: vehicle_state / lane_points / mission / driving_way_real / ctx
// output: reference_speed (LongitudinalControl로 전달됨)
//============================================
double PlanningNode::VelocityPlanning(const interface::VehicleState &vehicle_state, const interface::Lane &lane_points, const interface::Mission &mission, const interface::PolyfitLane &driving_way_real, const BehaviorContext &ctx) {
    /**
     * @brief Plan the desired speed along the given driving path
     * inputs: vehicle_state, lane_points, mission, driving_way_real, ctx
     * outputs: reference_speed
     * Purpose: Calculate desired speed based on curvature and driving mode
     */
    
     //perception_node.cpp에서 구현한 FindDrivingWay 함수 driving_way 사용

    // [12.01 다훈 수정] 기본 목표 속도 v_ref *0.5로 조정 (33.33*0.5=16.67m/s=60km/h)
    double v_ref = mission.speed_limit*1.0; // mission에서 직접 가져오기 (이게 v_lead)
    //=================================================
    //1) 곡률 기반 속도 제한 (v_kappa 계산)
    //=================================================
     //(1) PolyfitLaneData에서 a3, a2, a1, a0 가져옴 
    double a3 = driving_way_real.a3;
    double a2 = driving_way_real.a2;
    double a1 = driving_way_real.a1;
    double a0 = driving_way_real.a0;

    //(2) Kappa (곡률) 계산 및 속도 제한 적용 (ppt에서 k=2b=2*a2)
    double kappa = 2.0 * a2;

    //(3) 곡률 기반 속도 제한 v_kappa 계산 
    double eps = 1e-6; // 작은 값으로 나누기 방지
    double v_kappa;

    if(std::abs(kappa) < eps) {
        v_kappa = v_ref; // 곡률이 거의 0이면 그냥 목표 속도(limit_speed) 사용
    } else {
        v_kappa = std::sqrt(cfg_.param_max_lateral_accel / std::abs(kappa)); // v_kappa = sqrt(a_lat_max / |kappa|)
        v_kappa = std::min(v_kappa, v_ref); // 제한 속도 초과하지 않도록
    }

    //기본 속도 
    double reference_speed = std::min(v_ref, v_kappa);

    //=================================================
    // Mode 별 속도 결정
    //  - SCC: TTC 기반 속도 제어 (Spatial Buffer 적용) + SmoothSpeedProfile
    //  - LANE_CHANGE: 곡률 기반 속도 제한 + SmoothSpeedProfile
    //  - NORMAL_DRIVING: 기본 속도 + SmoothSpeedProfile
    //=================================================
    
    double target_speed = reference_speed;  // 기본 목표 속도
    
    //=================================================
    // [SCC in VelocityPlanning]: TTC 계산 및 속도 제어
    //=================================================
    if (ctx.current_mode == DrivingMode::SCC && ctx.has_dynamic_object) {
        //----------------------------------------
        // TTC 계산 (Spatial Buffer 적용)
        // spatial_buffer = T_gap * v_lead + D_min
        //----------------------------------------
        const double T_gap = 1.0;   // 안전 시간 간격 [초]
        const double D_min = 1.0;   // 최소 안전 거리 [m]
        
        double v_ego = vehicle_state.velocity;
        double v_lead = ctx.lead_velocity;
        double v_rel = v_ego - v_lead;  // 상대 속도 (v_rel>0: ego가 더 빠름)
        
        double spatial_buffer = T_gap * v_lead + D_min;
        double s_to_buffer = ctx.lead_s - spatial_buffer;   // Spatial Buffer까지 거리
        
        if (v_rel > 0 && s_to_buffer > 0) {                 // 추월 상황 (ego가 더 빠름) - TTC 계산
            double ttc = s_to_buffer / v_rel;

            if (ttc < 10.0) {                                // ttc<5초: 앞차 속도로 감속
                target_speed = std::min(v_lead, target_speed);
            } else {
                target_speed = reference_speed;             // 안전하면 기본 속도 유지
            }
        } else if (s_to_buffer <= 0) {
            // Spatial Buffer 내에 있음 - 강하게 감속
            target_speed = std::min(v_lead * 0.95, target_speed);  // 앞차 속도의 95%로 감속
            RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 500,
                "[SCC] Inside Spatial Buffer! spatial_buffer까지 거리(s_to_buffer): %.2f m ", s_to_buffer);
        }
    }
    //=================================================
    // [LANE_CHANGE in VelocityPlanning]: 차선 변경 시 속도 제어
    //=================================================
    else if (ctx.current_mode == DrivingMode::LANE_CHANGE || is_lane_changing_) {
        //-------------------------------------------------
        // 차선 변경 시 속도 제어
        //-------------------------------------------------
        double v_current = vehicle_state.velocity;
        
        // 차선 변경 시작 시 목표 속도 저장 (한 번만)
        static double lane_change_target_speed = 0.0;
        static bool lane_change_speed_set = false;
        
        if (!lane_change_speed_set || !is_lane_changing_) {
            lane_change_target_speed = v_current;
            
            // 최소/최대 속도 제한
            lane_change_target_speed = std::max(lane_change_target_speed, 8.0);   // 최소 8 m/s
            lane_change_target_speed = std::min(lane_change_target_speed, 20.0);  // 최대 20 m/s

            lane_change_speed_set = true;
        }
        
        // 목표 속도 = 저장된 감속 목표 (절대 현재 속도보다 높지 않음)
        target_speed = std::min(lane_change_target_speed, v_current);
        
        // 차선 변경 완료 시 리셋
        if (!is_lane_changing_) {
            lane_change_speed_set = false;
        }
    }
    // NORMAL_DRIVING 모드
    else {
        target_speed = reference_speed;
    }
    
    //=================================================
    // Forward-Backward Speed Profile Smoothing 적용
    // - 곡률 기반 속도 제한 + 가/감속 물리적 제한 적용
    // - SmoothSpeedProfile 내부에서 곡률 체크하므로 v_kappa는 이미 반영됨
    //=================================================
    reference_speed = SmoothSpeedProfile(driving_way_real, target_speed, vehicle_state.velocity);

    return reference_speed;
}

//============================================
// GlobalToLocal 함수 구현 
//============================================
std::pair<double, double> PlanningNode::GlobalToLocal(const interface::VehicleState &vehicle_state, double obj_x_global, double obj_y_global) {
    // A-1 obstacle (global coordinate) -> ego 기준 coordinate 변환
    double dx = obj_x_global - vehicle_state.x;
    double dy = obj_y_global - vehicle_state.y;
    double cos_yaw = std::cos(vehicle_state.yaw);
    double sin_yaw = std::sin(vehicle_state.yaw);
    double x_rel = cos_yaw * dx + sin_yaw * dy;
    double y_rel = -sin_yaw * dx + cos_yaw * dy;

    //x_rel: object의 global->local x좌표, y_rel: object의 global->local y좌표 (y_rel과 y_center비교해서 offset 판단)
    return std::make_pair(x_rel, y_rel); 
}

//============================================
//LocalToGlobal 함수 구현
//============================================
std::pair<double, double> PlanningNode::LocalToGlobal(const interface::VehicleState &vehicle_state, double x_rel, double y_rel) {
    double cos_yaw = std::cos(vehicle_state.yaw);
    double sin_yaw = std::sin(vehicle_state.yaw);

    double dx = cos_yaw * x_rel - sin_yaw * y_rel;
    double dy = sin_yaw * x_rel + cos_yaw * y_rel;
    double x_global = vehicle_state.x + dx;
    double y_global = vehicle_state.y + dy;

    return std::make_pair(x_global, y_global);
}

//============================================
// Frenet 좌표계 변환 함수 구현
//============================================

//============================================
// UpdateFrenetConverter 함수 구현
// - driving_way를 0.5m 간격으로 샘플링하여 FrenetConverter 초기화
//============================================
void PlanningNode::UpdateFrenetConverter(const interface::PolyfitLane &driving_way) {
    // 캐시된 값과 비교하여 변경되었는지 확인
    const double eps = 1e-9;
    if (frenet_converter_initialized_ &&
        std::abs(cached_driving_way_.a0 - driving_way.a0) < eps &&
        std::abs(cached_driving_way_.a1 - driving_way.a1) < eps &&
        std::abs(cached_driving_way_.a2 - driving_way.a2) < eps &&
        std::abs(cached_driving_way_.a3 - driving_way.a3) < eps) {
        return;  // 변경 없으면 업데이트 안함
    }
    
    // PolyfitLane을 s방향 0.5m 간격으로 샘플링 (FrenetConverter 생성자 인자로 사용)
    std::vector<double> x_pts, y_pts, psi_pts;
    
    // 샘플링 범위
    // x=0 (현재 차량 위치)부터 시작->s=0이 현재 차량 위치가 됨
    const double x_start = 0.0;  // 차량 위치부터 시작
    const double x_end = 40.0;   // 앞으로 40m까지
    const double ds_target = 0.5;  // 목표 s 간격 0.5m
    
    //----------------------------------------------
    //첫 번째 점 계산 (0,0) 
    double prev_x = x_start;
    double prev_y = driving_way.a3*pow(x_start,3) 
                    + driving_way.a2*pow(x_start,2) 
                    + driving_way.a1*x_start 
                    + driving_way.a0;
    x_pts.push_back(prev_x);
    y_pts.push_back(prev_y);
    
    // 첫 번째 점 heading(psi): dy/dx 계산
    double dy_dx = 3*driving_way.a3*pow(prev_x,2) 
                    + 2*driving_way.a2*prev_x 
                    + driving_way.a1;
    psi_pts.push_back(std::atan2(dy_dx, 1.0));
    //----------------------------------------------
    
    //------------------------------------------------
    //적분 스텝 (호장(driving_way)을 따라 0.5m 간격으로 샘플링)
    //x를 1m늘리면, 곡선을 따라 실제 길이 s는 몇 m늘어나는가 
    double x = x_start;
    while (x < x_end) {
        // 현재 점에서의 ds/dx = sqrt(1 + (dy/dx)^2)
        dy_dx = 3*driving_way.a3*pow(x,2) + 2*driving_way.a2*x + driving_way.a1;
        double ds_dx = std::sqrt(1.0 + dy_dx * dy_dx); //x가 늘어날 때 s는 얼만큼 늘어나는지 
        
        // dx = ds / (ds/dx) ≈ 0.1 / ds_dx
        double dx = ds_target / ds_dx; //현재 곡률에서 s를 0.5늘릴 때 필요한 x변화량
        x += dx;
        
        if (x > x_end) break;
        
        // 새 점 계산
        double y = driving_way.a3*pow(x,3) + driving_way.a2*pow(x,2) 
                 + driving_way.a1*x + driving_way.a0;
        
        x_pts.push_back(x);
        y_pts.push_back(y);
        
        // heading 계산 (접선(tangent) 벡터)
        dy_dx = 3*driving_way.a3*pow(x,2) + 2*driving_way.a2*x + driving_way.a1;
        psi_pts.push_back(std::atan2(dy_dx, 1.0));
    }
    //------------------------------------------------
    
    // FrenetConverter: (x,y)<->(s,d) 변환 클래스
    frenet_converter_ = FrenetConverter(x_pts, y_pts, psi_pts, false);
    
    // 캐시 업데이트
    cached_driving_way_ = driving_way;
    frenet_converter_initialized_ = true;
}

PlanningNode::FrenetCoordinate PlanningNode::CartesianToFrenet(double x_rel, double y_rel, const interface::PolyfitLane &driving_way) {
    FrenetCoordinate frenet;
    
    // FrenetConverter가 없으면 업데이트 
    if (!frenet_converter_initialized_) {
        UpdateFrenetConverter(driving_way);
    }
    
    // FrenetConverter 사용
    auto [s, d] = frenet_converter_.cartesian_to_frenet(x_rel, y_rel);
    frenet.s = s;
    frenet.d = d;
    
    return frenet;
}

//============================================
//FrenetToCartesian 함수 구현 (Frenet -> Ego 좌표계 변환)
//============================================
std::pair<double, double> PlanningNode::FrenetToCartesian(double s, double d, const interface::PolyfitLane &driving_way) {
    // FrenetConverter가 없으면 업데이트
    if (!frenet_converter_initialized_) {
        UpdateFrenetConverter(driving_way);
    }
    
    // FrenetConverter 사용
    return frenet_converter_.frenet_to_cartesian(s, d);
}
//============================================
// LaneChange 함수 구현 (Frenet 경계조건 기반 3차 다항식)
// 
// ##아이디어
// Frenet 좌표계에서 4개 경계조건으로 3차 다항식 계수를 해석적으로 계산
// 
// ##경계 조건
// 1. d(0) = d0        : 시작점 위치 (현재 d, 보통 0)
// 2. d'(0) = 0        : 시작점 기울기 (현재 진행방향과 평행)
// 3. d(sf) = df       : 끝점 위치 (목표 d, ±4m)
// 4. d'(sf) = 0       : 끝점 기울기 (목표 차선과 평행하게 진입)
//
// 3차 다항식: d(s) = a0 + a1*s + a2*s² + a3*s³
// 해석적 해:
//   a0 = d0
//   a1 = 0
//   a2 = 3*(df - d0) / sf²
//   a3 = -2*(df - d0) / sf³
//
// 장점:
// - S자 없이 부드러운 단일 곡선
// - 시작/끝 접선 연속 (급격한 조향 방지)
// - 도로 곡률과 독립적인 경로 생성
//============================================
interface::PolyfitLane PlanningNode::LaneChange(const interface::VehicleState &vehicle_state, const interface::PolyfitLane &driving_way, const BehaviorContext &ctx) {
    
    const double LANE_WIDTH = 4.0;
    interface::PolyfitLane result;
    result.frame_id = driving_way.frame_id;
    
    //=================================================================
    // 1) 차선 변경 시작: Frenet 경계조건 설정 → Global 경로 저장
    //=================================================================
    if (!lane_change_path_saved_) {
        // 목표 차선 선택 (왼쪽 우선, 단 막혀있으면 오른쪽)
        int target_lane_id = current_lane_id_;
        bool lane_found = false;
        
        // 1순위: 왼쪽 (Lane 2)
        if (current_lane_id_ < 2 && !ctx.left_lane_blocked) {
            target_lane_id = current_lane_id_ + 1;
            lane_found = true;
        }
        // 2순위: 오른쪽 (Lane 0)
        else if (current_lane_id_ > 0 && !ctx.right_lane_blocked) {
            target_lane_id = current_lane_id_ - 1;
            lane_found = true;
        }
        // 사용 가능한 차선이 없으면 원래 경로 유지
        if (!lane_found) {
            RCLCPP_WARN(this->get_logger(), "[LaneChange] ❌ No available lane! All lanes blocked.");
            is_lane_changing_ = false;  // 차선 변경 중단
            return driving_way;
        }
        
        target_lane_id_ = target_lane_id;
        
        //=============================================================
        // Frenet 경계 조건 설정
        //=============================================================
        double d0 = 0.0;  // 시작 d (현재 차선 중앙)
        double df = (target_lane_id - current_lane_id_) * LANE_WIDTH;  // 목표 d (+4 또는 -4)
        
        // 장애물까지 거리를 sf로 사용 (단, 최소 거리 보장)
        double obstacle_s = ctx.static_object_s;    // = frenet.s
        const double LANE_CHANGE_MARGIN = 1.0;  // 최소 차선 변경 거리
        double sf = obstacle_s - LANE_CHANGE_MARGIN;  // 장애물 1m 전에 완료

        RCLCPP_INFO(this->get_logger(), 
            "[LaneChange] START! Frenet: d0=%.1f → df=%.1f, sf=%.1f m (obstacle_s=%.1f)",
            d0, df, sf, obstacle_s);
        
        //=============================================================
        // 3차 다항식 계수 해석적 계산 (Frenet d(s))
        // d(s) = a0 + a1*s + a2*s² + a3*s³
        //=============================================================
        double frenet_a0 = d0;                                          // d(0) = d0
        double frenet_a1 = 0.0;  // d'(0) = 0                           // d'(0) = 0 
        double frenet_a2 = 3.0 * (df - d0) / (sf * sf);                 // d(sf) = df    
        double frenet_a3 = -2.0 * (df - d0) / (sf * sf * sf);           // d'(sf) = 0
        
        //=============================================================
        // Frenet 경로점 생성 → Local → Global 변환하여 저장
        // sf까지만 생성 (이후는 driving_way 그대로 사용)
        //=============================================================
        lane_change_path_global_.clear();
        
        const double ds = 1.0;  // 1m 간격으로 샘플링
        for (double s = 0.0; s <= sf; s += ds) {
            // Frenet d(s) 계산 (3차 다항식)
            double d = frenet_a0 + frenet_a1 * s + frenet_a2 * s * s + frenet_a3 * s * s * s;
            
            // Frenet → Local (FrenetConverter 사용)
            auto [x_local, y_local] = FrenetToCartesian(s, d, driving_way);
            
            // Local → Global
            auto [x_global, y_global] = LocalToGlobal(vehicle_state, x_local, y_local);
            
            lane_change_path_global_.push_back({x_global, y_global});
        }
        
        // 목표점 (sf, df) Global 좌표 저장 (빨간 구체용)
        auto [target_x_local, target_y_local] = FrenetToCartesian(sf, df, driving_way);
        auto [gx, gy] = LocalToGlobal(vehicle_state, target_x_local, target_y_local);
        target_point_global_x_ = gx;
        target_point_global_y_ = gy;
        
        // sf 저장 (완료 조건용)
        lane_change_sf_ = sf;
        
        lane_change_path_saved_ = true;
        is_lane_changing_ = true;
        lane_change_counter_ = 0;
        
        RCLCPP_INFO(this->get_logger(), 
            "[LaneChange] Path created! %zu points, frenet: a2=%.6f, a3=%.9f",
            lane_change_path_global_.size(), frenet_a2, frenet_a3);
    }
    
    lane_change_counter_++;
    
    //=================================================================
    // 2) Global 경로점들 → Local 변환
    //=================================================================
    std::vector<double> x_local_points, y_local_points;
    
    const double ROI_MAX = 15.0; // 15m 이내 점들만 사용

    for (const auto& [gx, gy] : lane_change_path_global_) {
        auto [lx, ly] = GlobalToLocal(vehicle_state, gx, gy);
        
        // 차량 앞쪽 점만 사용 (x > -1m)
        if (lx > -1.0 && lx < ROI_MAX) {
            x_local_points.push_back(lx);
            y_local_points.push_back(ly);
        }
    }
    
    // 빨간 구체 Local 좌표
    auto [target_x_local, target_y_local] = GlobalToLocal(vehicle_state, 
                                                           target_point_global_x_, 
                                                           target_point_global_y_);
    
    //=================================================================
    // 3) 완료 조건: 목표점을 지나갔거나 타임아웃
    // ✅ 수정: 목표점 통과 시 바로 완료 (원래 차선으로 복귀하지 않음)
    //=================================================================
    bool passed_target = (target_x_local < 0.0);  // 목표점이 차량 뒤로 가면 완료
    bool timeout = (lane_change_counter_ > 1000);  // 10초
    bool no_points = (x_local_points.size() < 4);
    
    if (passed_target || timeout || no_points) {
        current_lane_id_ = target_lane_id_;
        is_lane_changing_ = false;
        lane_change_path_saved_ = false;
        lane_change_path_global_.clear();
        
        RCLCPP_INFO(this->get_logger(), 
            "[LaneChange] ✅ DONE! new_lane=%d (passed=%d, timeout=%d, no_points=%d)",
            current_lane_id_, passed_target, timeout, no_points);
        return driving_way;
    }
    
    //=================================================================
    // 4) Local 점들로 Polynomial Fitting (3차)
    //=================================================================
    int n = x_local_points.size();
    
    Eigen::MatrixXd A(n, 4);
    Eigen::VectorXd b(n);
    
    for (int i = 0; i < n; i++) {
        double x = x_local_points[i];
        A(i, 0) = 1.0;
        A(i, 1) = x;
        A(i, 2) = x * x;
        A(i, 3) = x * x * x;
        b(i) = y_local_points[i];
    }
    
    Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    
    result.a0 = coeffs(0);
    result.a1 = coeffs(1);
    result.a2 = coeffs(2);
    result.a3 = coeffs(3);
    
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 500,
        "[LaneChange] target_local=(%.1f,%.1f), points=%d, a0=%.3f, a1=%.3f, a2=%.6f",
        target_x_local, target_y_local, n, result.a0, result.a1, result.a2);
    
    //=================================================================
    // 5) 시각화: 빨간 구체 (목표점)
    //=================================================================
    visualization_msgs::msg::Marker target_marker;
    target_marker.header.frame_id = driving_way.frame_id;
    target_marker.header.stamp = this->now();
    target_marker.ns = "lane_change_target";
    target_marker.id = 0;
    target_marker.type = visualization_msgs::msg::Marker::SPHERE;
    target_marker.action = visualization_msgs::msg::Marker::ADD;
    target_marker.pose.position.x = target_x_local;
    target_marker.pose.position.y = target_y_local;
    target_marker.pose.position.z = 0.5;
    target_marker.pose.orientation.w = 1.0;
    target_marker.scale.x = 1.5;
    target_marker.scale.y = 1.5;
    target_marker.scale.z = 1.5;
    target_marker.color.r = 1.0;
    target_marker.color.g = 0.0;
    target_marker.color.b = 0.0;
    target_marker.color.a = 0.9;
    p_lane_change_target_->publish(target_marker);
    
    //=================================================================
    // 6) 시각화: 실제 다항식 경로 (초록 선) + 경로점들 (파란 점)
    //=================================================================
    visualization_msgs::msg::MarkerArray path_markers;
    
    // 6-1) 실제 fitting된 다항식 경로 (초록 선) - 이게 실제 주행 경로
    visualization_msgs::msg::Marker poly_line;
    poly_line.header.frame_id = driving_way.frame_id;
    poly_line.header.stamp = this->now();
    poly_line.ns = "lane_change_polynomial";
    poly_line.id = 101;
    poly_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    poly_line.action = visualization_msgs::msg::Marker::ADD;
    
    // ✅ 수정: target point까지만 시각화 (x_local_points 범위 사용)
    // 다항식을 따라 점 생성 (차량 앞쪽 경로만)
    if (!x_local_points.empty()) {
        double x_max = x_local_points.back();  // 마지막 점까지만
        for (double x = 0.0; x <= x_max; x += 0.5) {
            geometry_msgs::msg::Point p;
            p.x = x;
            p.y = result.a0 + result.a1 * x + result.a2 * x * x + result.a3 * x * x * x;
            p.z = 0.15;
            poly_line.points.push_back(p);
        }
    }
    poly_line.scale.x = 0.15;
    poly_line.color.g = 1.0;  // 초록색
    poly_line.color.a = 1.0;
    path_markers.markers.push_back(poly_line);
    
    // 6-2) Frenet 경로점들 (파란 점) - 참고용
    visualization_msgs::msg::Marker path_points;
    path_points.header.frame_id = driving_way.frame_id;
    path_points.header.stamp = this->now();
    path_points.ns = "lane_change_points";
    path_points.id = 102;
    path_points.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    path_points.action = visualization_msgs::msg::Marker::ADD;
    
    for (size_t i = 0; i < x_local_points.size(); i++) {
        geometry_msgs::msg::Point p;
        p.x = x_local_points[i];
        p.y = y_local_points[i];
        p.z = 0.2;
        path_points.points.push_back(p);
    }
    path_points.scale.x = 0.3;
    path_points.scale.y = 0.3;
    path_points.scale.z = 0.3;
    path_points.color.b = 1.0;  // 파란색
    path_points.color.a = 0.7;
    path_markers.markers.push_back(path_points);
    
    p_lane_change_path_->publish(path_markers);
    
    return result;
}
int main(int argc, char **argv) {
    std::string node_name = "planning_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlanningNode>(node_name));
    rclcpp::shutdown();
    return 0;
}
