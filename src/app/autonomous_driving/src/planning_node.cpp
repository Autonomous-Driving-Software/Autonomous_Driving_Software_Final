/*
 * planning_node.cpp
 */
#include "autonomous_driving_config.hpp"
#include "planning_node.hpp"
#include "frenet_converter.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <rviz_2d_overlay_msgs/msg/overlay_text.hpp>

using namespace std;
/*
#1. PlanningNode 생성*/
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

    /* 노드를 ROS 네트워크에 연결하는 부분: create_subscription<메시지타입>(토픽이름, QoS, callback)*/
    //===========================
    //subscriber init
    //===========================
    s_manual_input_ = this->create_subscription<ad_msgs::msg::VehicleCommand>(
        "/manual_input", qos_profile, std::bind(&PlanningNode::CallbackManualInput, this, std::placeholders::_1));

    //(2) s_vehicle_state_
    s_vehicle_state_ = 
    this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&
        PlanningNode::CallbackVehicleState, this, 
    std::placeholders::_1));

    //(3) s_lane_points_
    s_lane_points_ = 
    this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&PlanningNode::CallbackLanePoints, this, std::placeholders::_1));
        
    //(4) s_mission_
    s_mission_ = this->create_subscription<ad_msgs::msg::Mission>(
        "mission", qos_profile, std::bind(&PlanningNode::CallbackMission, this, std::placeholders::_1));

    //(5) s_driving_way_
    s_driving_way_ = this->create_subscription<ad_msgs::msg::PolyfitLaneData>(
        "driving_way", qos_profile, std::bind(&PlanningNode::CallbackPolyfitLaneData, this, std::placeholders::_1));

    //============================
    //publisher init
    //============================
    p_vehicle_command_ = this->create_publisher<ad_msgs::msg::VehicleCommand>(
        "vehicle_command", qos_profile);
    p_driving_way_real_ = this->create_publisher<ad_msgs::msg::PolyfitLaneData>(
        "driving_way_real", qos_profile);
    p_driving_way_points_ = this->create_publisher<ad_msgs::msg::LanePointData>(
        "driving_way_points", qos_profile);

    //[다훈 추가]p_reference_speed_ (이게 맞나?)(lon에 넘기려고)
    p_reference_speed_ = this->create_publisher<std_msgs::msg::Float32>(
        "reference_speed", qos_profile);
    
    // Debug Publishers (for PlotJuggler)
    p_d_range_ = this->create_publisher<std_msgs::msg::Float32>(
        "debug/d_range", qos_profile);
    p_safe_distance_ = this->create_publisher<std_msgs::msg::Float32>(
        "debug/safe_distance", qos_profile);
    
    // Visualization Publishers
    p_lane_change_target_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "lane_change_target", qos_profile);
    p_lane_change_path_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "lane_change_path", qos_profile);
    p_lane_id_text_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "lane_id_text", qos_profile);
    p_info_text_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "planning_info_text", qos_profile);
    // 화면 고정 오버레이 텍스트 publisher (RViz 우측 상단에 고정)
    p_overlay_text_ = this->create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>(
        "planning_overlay_text", qos_profile);
    p_mu_estimated_ = this->create_publisher<std_msgs::msg::Float32>("mu_estimated", qos_profile);
    p_a_lat_measured_ = this->create_publisher<std_msgs::msg::Float32>("a_lat_measured", qos_profile);

    // Initialize
    Init(this->now());

    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); }); 

}
PlanningNode::~PlanningNode() {}

void PlanningNode::Init(const rclcpp::Time &current_time) { //현재 아무것도 안함
    (void)current_time;
}

void PlanningNode::ProcessParams() {
    //get parameters(파라미터 값 가져오기) + 값 읽어서 cfg_에 저장
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("autonomous_driving/use_manual_inputs", cfg_.use_manual_inputs);
}

void PlanningNode::Run() {
    auto current_time = this->now(); //타이머로 Run() 반복 호출
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "Running ..."); //로그는 최소 1000ms에 한번만 출력
    ProcessParams(); //바로 위에 선언되어 있고, Run함수에서 호출해서 동적으로 parameters 업데이트

    //=======================================================================
    // input 유효성 체크(아직 필요한 input 토픽 안들어왔으면 알고리즘 실행하지 않고 기다림)
    //=======================================================================
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

    //=======================================================================
    // 1. <Get subscribe variables>
    // [일종의 input데이터 수집 단계 (멤버 변수 -> 지역변수로 복사 (mutex로 보호))]
    //=======================================================================
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
    // <기본 Polyfit을 포인트 경로로 변경> (ego 기준 local좌표계 (x_local, y_local))
    //===================================================
    interface::Lane base_path_points = SamplePathFromPolyfit(driving_way);

    //===================================================
    // <FrenetConverter 업데이트> (포인트 기반)
    //===================================================
    UpdateFrenetConverter(base_path_points);
    
    // 원래 차선 경로도 별도로 저장 (차선 변경 완료 확인용)
    if (!is_lane_changing_) {
        base_frenet_converter_ = frenet_converter_;
        base_frenet_converter_initialized_ = frenet_converter_initialized_;
    }

    //===================================================
    // 3.[11.28 다훈 수정] <Behavior Planning>
    //===================================================
    PlanningNode::BehaviorContext ctx = BehaviorPlanning(vehicle_state, mission, base_path_points);

    // 모드 확인 로그
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, 
        "[Run] Current Mode: %d, has_static: %d, is_changing: %d", 
        static_cast<int>(ctx.current_mode), ctx.has_static_object, is_lane_changing_);

    //Lane Change 모드일 때 새로운 경로 생성 (Control로 넘어가는건 driving_way_points_)
    interface::PolyfitLane driving_way_real = driving_way; // 호환 퍼블리시용
    driving_way_points_.point.clear();
    
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

        driving_way_points_ = LaneChange(vehicle_state, driving_way, ctx);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000,
            "[Run] LaneChange path points: %zu pts", driving_way_points_.point.size());
    } else {
        driving_way_points_ = base_path_points;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000,
            "[Run] Normal path points: %zu pts", driving_way_points_.point.size());
    }

    // 최신 경로 기준 변환기 동기화 (차선 변경 포함)
    UpdateFrenetConverter(driving_way_points_);

    //[다훈 수정2] 포인트 기반 경로로 속도 계획
    double reference_speed = VelocityPlanning(vehicle_state, lane_points, mission, driving_way_points_, ctx);

    //===================================================
    // Visualization: 화면 고정 오버레이 텍스트 (RViz 우측 상단)
    // - rviz_2d_overlay_msgs::msg::OverlayText 사용
    // - 줌인/줌아웃해도 크기 변하지 않고 화면에 고정됨
    //===================================================
    auto mode_to_string = [](DrivingMode mode) {
        switch (mode) {
            case DrivingMode::LANE_KEEPING: return std::string("LANE_KEEPING");
            case DrivingMode::SCC:          return std::string("SCC");
            case DrivingMode::LANE_CHANGE:  return std::string("LANE_CHANGE");
            default:                        return std::string("UNKNOWN");
        }
    };

    // OverlayText 메시지 생성 (화면 고정)
    rviz_2d_overlay_msgs::msg::OverlayText overlay_msg;
    overlay_msg.action = rviz_2d_overlay_msgs::msg::OverlayText::ADD;
    
    // 텍스트 박스 크기 설정
    overlay_msg.width = 500;   // 박스 너비 (픽셀)
    overlay_msg.height = (ctx.current_mode == DrivingMode::SCC) ? 270 : 130;  // SCC일 때 더 큰 박스
    
    // 화면 우측 상단에 배치
    overlay_msg.horizontal_alignment = rviz_2d_overlay_msgs::msg::OverlayText::RIGHT;
    overlay_msg.vertical_alignment = rviz_2d_overlay_msgs::msg::OverlayText::TOP;
    overlay_msg.horizontal_distance = 20;  // 우측 가장자리에서 20픽셀 떨어짐
    overlay_msg.vertical_distance = 20;    // 상단 가장자리에서 20픽셀 떨어짐
    
    // 배경색 설정 (반투명 검정)
    overlay_msg.bg_color.r = 0.0f;
    overlay_msg.bg_color.g = 0.0f;
    overlay_msg.bg_color.b = 0.0f;
    overlay_msg.bg_color.a = 0.0f;  // 70% 불투명
    
    // 텍스트 색상 설정 (밝은 초록)
    overlay_msg.fg_color.r = 0.1f;
    overlay_msg.fg_color.g = 1.0f;
    overlay_msg.fg_color.b = 0.1f;
    overlay_msg.fg_color.a = 1.0f;
    
    // 폰트 설정
    overlay_msg.text_size = 14.0f;      // 폰트 크기 (포인트)
    overlay_msg.line_width = 2;          // 테두리 두께
    overlay_msg.font = "DejaVu Sans Mono";  // 고정폭 폰트 사용
    
    // 텍스트 내용 생성
    std::ostringstream info_text;
    info_text << "===== Planning Info =====\n"
              << "Mode: " << mode_to_string(ctx.current_mode) << "\n"
              << std::fixed << std::setprecision(2)
              << "Target Speed: " << reference_speed << " m/s\n"
              << "Ego Velocity: " << vehicle_state.velocity << " m/s\n"
              << "Lane ID: " << std::to_string(ctx.current_lane_id) << "\n";
    
    if (mission.road_condition == "Ice") {
        info_text << "Road Condition: Ice\n";
    }
    else{
        info_text << "Road Condition: \n";
    }
    
    // SCC 모드일 때 safe distance정보 추가
    if (ctx.current_mode == DrivingMode::SCC && ctx.has_dynamic_object) {
        double v_rel = vehicle_state.velocity - ctx.lead_velocity;
        const double d_range = ctx.lead_s - 1.0;
        double ttc = (v_rel > 0.01) ? (d_range / v_rel) : 999.0;
        const double T_gap = 0.1; //안전 시간 간격 [s]
        const double D_min = 10.0; //최소 안전 거리 [m]
        double safe_distance = T_gap*vehicle_state.velocity + D_min;
        
        info_text << "\n----- SCC Info -----\n"
                  << "Distance from ego to dynamic: " << d_range << " m\n"
                  << "Safe Distance: " << safe_distance << " m\n"
                  << "TTC: " << ttc << " s";
    }
    
    overlay_msg.text = info_text.str();
    
    // 오버레이 텍스트 publish
    p_overlay_text_->publish(overlay_msg);
    //===================================================
    // Publish output
    //===================================================
    //Publish driving way
    p_driving_way_real_->publish(ros2_bridge::UpdatePolyfitLane(driving_way_real));

    //[다훈 수정] reference_speed publish
    std_msgs::msg::Float32 ref_msg;
    ref_msg.data = reference_speed;
    p_reference_speed_->publish(ref_msg);

    // 차선 변경 Frenet 경계 조건이 반영된 실제 경로 점들 Publish (control에서 직접 사용)
    if (!driving_way_points_.point.empty()) {
        p_driving_way_points_->publish(ros2_bridge::UpdateLanePoints(driving_way_points_, cfg_.vehicle_namespace));
    } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 1000,
            "[Run] driving_way_points_ empty, skip publish");
    }
}



//============================================
//#BehaviorPlanning 함수 구현
// input: vehicle_state / mission / driving_way
// output: BehaviorContext (현재 모드 정보) / SCC일 때 TTC에 맞는 v_lead / Lane Change일 때 driving_way_real
//============================================
PlanningNode::BehaviorContext PlanningNode::BehaviorPlanning(const interface::VehicleState &vehicle_state, const interface::Mission &mission, const interface::Lane &driving_path) {
    // ctx initialization
    BehaviorContext ctx;
    ctx.current_mode = DrivingMode::LANE_KEEPING; // 기본 모드 설정
    ctx.mission = mission; // mission 정보 저장
    (void)driving_path; // 현재는 변환기 상태만 사용
    
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
        FrenetCoordinate frenet = CartesianToFrenet(x_rel, y_rel);

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
        ctx.current_mode = DrivingMode::LANE_KEEPING;
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
double PlanningNode::SmoothSpeedProfile(const interface::Lane &driving_path_points, double v_ref, double current_velocity) {
    // 파라미터
    const double a_max = cfg_.param_a_max;   // 최대 가속도 [m/s²] (양수: 2.0)
    const double a_min = cfg_.param_a_min;   // 최대 감속도 [m/s²] (음수: -3.0)

    if (driving_path_points.point.size() < 2) {
        return v_ref;
    }

    std::vector<double> s_arr; // 경로를 s(누적거리)축으로 변환 + 초기 속도 배열 만들기 
    std::vector<double> v_arr;
    s_arr.reserve(driving_path_points.point.size());
    v_arr.reserve(driving_path_points.point.size());
    double s_acc = 0.0;
    s_arr.push_back(0.0);
    v_arr.push_back(v_ref);

    for (size_t i = 1; i < driving_path_points.point.size(); ++i) {
        const auto &p_prev = driving_path_points.point[i - 1];
        const auto &p_cur = driving_path_points.point[i];
        double ds = std::hypot(p_cur.x - p_prev.x, p_cur.y - p_prev.y);
        s_acc += ds;
        s_arr.push_back(s_acc);
        v_arr.push_back(v_ref);
    }

    // 곡률 기반 속도 제한
    const double eps = 1e-6;
    for (size_t i = 1; i + 1 < driving_path_points.point.size(); ++i) {
        const auto &p0 = driving_path_points.point[i - 1];
        const auto &p1 = driving_path_points.point[i];
        const auto &p2 = driving_path_points.point[i + 1];

        double a = std::hypot(p1.x - p0.x, p1.y - p0.y);
        double b = std::hypot(p2.x - p1.x, p2.y - p1.y);
        double c = std::hypot(p2.x - p0.x, p2.y - p0.y);
        if (a < eps || b < eps || c < eps) {
            continue;
        }
        double area2 = std::abs((p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x));
        double kappa = (2.0 * area2) / (a * b * c);
        if (kappa > eps) {
            double v_kappa = std::sqrt(cfg_.param_max_lateral_accel / kappa);
            v_arr[i] = std::min(v_arr[i], v_kappa);
        }
    }

    // Forward Pass: 가속 제한 (a_max)
    for (size_t i = 1; i < v_arr.size(); ++i) {
        double ds = s_arr[i] - s_arr[i - 1];
        double v0 = v_arr[i - 1];
        double v_target = v_arr[i];
        double v_max_from_prev_sq = v0 * v0 + 2.0 * a_max * ds;
        double v_max_from_prev = (v_max_from_prev_sq > 0.0) ? std::sqrt(v_max_from_prev_sq) : 0.0;
        v_arr[i] = std::min(v_target, v_max_from_prev);
    }

    // Backward Pass: 감속 제한 (a_min, 음수)
    for (int i = static_cast<int>(v_arr.size()) - 2; i >= 0; --i) {
        double ds = s_arr[i + 1] - s_arr[i];
        double v_next = v_arr[i + 1];
        double v_max_sq = v_next * v_next - 2.0 * a_min * ds;  // a_min은 음수
        double v_max_from_next = (v_max_sq > 0.0) ? std::sqrt(v_max_sq) : 0.0;
        v_arr[i] = std::min(v_arr[i], v_max_from_next);
    }

    double smoothed_speed = v_arr.front();

    double ds0 = (v_arr.size() > 1) ? (s_arr[1] - s_arr[0]) : 0.5;
    double v_max_accel = std::sqrt(current_velocity * current_velocity + 2.0 * a_max * ds0);
    smoothed_speed = std::min(smoothed_speed, v_max_accel);

    if (current_velocity < 0.1 && smoothed_speed < 0.5) {
        smoothed_speed = std::min(v_ref, 5.0);
    }

    return smoothed_speed;
}

//============================================
// Polyfit 경로를 일정 간격 포인트로 샘플링 (Control에서 재사용)
//============================================
interface::Lane PlanningNode::SamplePathFromPolyfit(const interface::PolyfitLane &lane, double max_s, double ds) {
    interface::Lane path;
    path.frame_id = lane.frame_id;
    path.id = lane.id;

    if (ds <= 0.0) {
        ds = 0.5;
    }

    for (double x = 0.0; x <= max_s; x += ds) {
        interface::Point2D pt;
        pt.x = x;
        pt.y = lane.a3 * pow(x, 3) + lane.a2 * pow(x, 2) + lane.a1 * x + lane.a0;
        path.point.push_back(pt);
    }

    // max_s가 ds 간격과 맞지 않을 때 마지막 점 보정
    if (path.point.empty() || path.point.back().x < max_s - 1e-3) {
        interface::Point2D pt;
        pt.x = max_s;
        pt.y = lane.a3 * pow(max_s, 3) + lane.a2 * pow(max_s, 2) + lane.a1 * max_s + lane.a0;
        path.point.push_back(pt);
    }

    return path;
}

//============================================
// 로컬 좌표계 포인트 벡터를 interface::Lane 형태로 래핑
//============================================
interface::Lane PlanningNode::BuildLaneFromLocalPoints(const std::vector<double> &x_pts, const std::vector<double> &y_pts, const std::string &frame_id, const std::string &id) {
    interface::Lane lane;
    lane.frame_id = frame_id;
    lane.id = id;

    size_t n = std::min(x_pts.size(), y_pts.size());
    lane.point.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        interface::Point2D pt;
        pt.x = x_pts[i];
        pt.y = y_pts[i];
        lane.point.push_back(pt);
    }

    return lane;
}

//============================================
// VelocityPlanning 
// input: vehicle_state / lane_points / mission / driving_way_real / ctx
// output: reference_speed (LongitudinalControl로 전달됨)
//============================================
double PlanningNode::VelocityPlanning(const interface::VehicleState &vehicle_state, const interface::Lane &lane_points, const interface::Mission &mission, const interface::Lane &driving_path_points, const BehaviorContext &ctx) {
    /**
     * @brief Plan the desired speed along the given driving path
     * inputs: vehicle_state, lane_points, mission, driving_way_real, ctx
     * outputs: reference_speed
     * Purpose: Calculate desired speed based on curvature and driving mode
     */
    (void)lane_points; // 현재 곡률 계산은 driving_path_points 기반

    double v_ref = mission.speed_limit * 1.0; // mission에서 직접 가져오기

    //=================================================
    // 최대 횡가속도 계산
    //=================================================
    //1. 일단 mu 실험적으로 구하기 
    double a_lat_max = cfg_.param_max_lateral_accel; //최대 횡가속도 기본 
    double mu = 0.2; // 타이어-노면 마찰계수 
    double g = 9.81; // 중력 가속도
    
    if (mission.road_condition == "Ice") {
        a_lat_max = mu * g;
    }
    
    //=================================================
    //[실험용] mu 추정 및 publish
    //=================================================
    double a_lat_measured = std::abs(vehicle_state.velocity * vehicle_state.yaw_rate);
    double mu_estimated = a_lat_measured / 9.81;
    mu_estimated = std::min(mu_estimated, 1.0);

    // Publish for PlotJuggler
    std_msgs::msg::Float32 mu_msg;
    mu_msg.data = static_cast<float>(mu_estimated);
    p_mu_estimated_->publish(mu_msg);

    std_msgs::msg::Float32 a_lat_msg;
    a_lat_msg.data = static_cast<float>(a_lat_measured);
    p_a_lat_measured_->publish(a_lat_msg);
    //=================================================
    //1) 곡률 최대값 계산 → 속도 제한 (point 기반 곡률 계산)
    //     - 곡률 계산 방법: 3점 곡률 계산식 사용 
    //=================================================
    double max_kappa = 0.0;
    const size_t n_pts = driving_path_points.point.size(); 
    for (size_t i = 1; i + 1 < n_pts; ++i) {
        const auto &p0 = driving_path_points.point[i - 1];     //이전 포인트 (i-1)
        const auto &p1 = driving_path_points.point[i];         //현재 포인트 (i)
        const auto &p2 = driving_path_points.point[i + 1];     //다음 포인트 (i+1)

        double a = std::hypot(p1.x - p0.x, p1.y - p0.y);     //이전 포인트와 현재 포인트 사이의 길이 //hypot(=유클리드 거리)
        double b = std::hypot(p2.x - p1.x, p2.y - p1.y);     //현재 포인트와 다음 포인트 사이의 길이
        double c = std::hypot(p2.x - p0.x, p2.y - p0.y);     //이전 포인트와 다음 포인트 사이의 길이

        if (a < 1e-4 || b < 1e-4 || c < 1e-4) {              //점들이 겹치면 곡률 계산식 분모가 0에 가까워져서 발산->skip
            continue;
        }

        double area2 = std::abs((p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)); 
        double kappa = (2.0 * area2) / (a * b * c);
        max_kappa = std::max(max_kappa, kappa);
    }

    const double eps = 1e-6;
    double v_kappa = v_ref;
    if (max_kappa > eps) {
        v_kappa = std::sqrt(a_lat_max / max_kappa);
        v_kappa = std::min(v_kappa, v_ref);
    }

    double reference_speed = std::min(v_ref, v_kappa);

    //=================================================
    // Mode 별 속도 결정
    //  - SCC: TTC 기반 속도 제어 (safe distance적용) + SmoothSpeedProfile
    //  - LANE_CHANGE: 곡률 기반 속도 제한 + SmoothSpeedProfile
    //  - LANE_KEEPING: 기본 속도 + SmoothSpeedProfile
    //=================================================
    
    double target_speed = reference_speed;  // 기본 목표 속도
    
    //=================================================
    // [SCC in VelocityPlanning]: TTC 계산 및 속도 제어
    //=================================================
    if (ctx.current_mode == DrivingMode::SCC && ctx.has_dynamic_object) { // has_dynamic_object가 들어오는 순간 20m안쪽임. 
        //----------------------------------------
        // TTC 계산 (safe distance적용)
        // 
        //----------------------------------------
        const double TTC_THRESHOLD = 2.0; //유지하고 싶은 TTC [s]
        const double T_gap = 0.1; //안전 시간 간격 [s]
        const double D_min = 10.0; //최소 안전 거리 [m]

        double v_ego = vehicle_state.velocity;
        double v_lead = ctx.lead_velocity;
        double v_rel = v_ego - v_lead;
        double d_range = ctx.lead_s-1.0; 
        double safe_distance = T_gap *v_ego + D_min;
        double s_to_buffer = d_range - safe_distance;
        double ttc = d_range / v_rel;       //이중제약 (ttc<2이면 완전 급제동)

        // Publish d_range and safe_distance for PlotJuggler
        std_msgs::msg::Float32 d_range_msg;
        d_range_msg.data = static_cast<float>(d_range);
        p_d_range_->publish(d_range_msg);
        
        std_msgs::msg::Float32 safe_distance_msg;
        safe_distance_msg.data = static_cast<float>(safe_distance);
        p_safe_distance_->publish(safe_distance_msg);

        if (v_rel > 0) {
            if (ttc < TTC_THRESHOLD) {
                target_speed = std::min(v_lead*0.2, target_speed); //TTC는 급제동용 (충돌 방지)
            }
            else {
                if (s_to_buffer > 0) {       
                    double v_safe = v_lead + s_to_buffer;   //안전거리 아직 잘 유지 중이면 
                    target_speed = std::min(v_safe, target_speed);
                    //target_speed = reference_speed;
                }
                else { //s_to_buffer <= 0 안전거리 안쪽으로 들어오면
                    target_speed = std::min(v_lead - s_to_buffer, target_speed);
                }
            }
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
    // LANE_KEEPING 모드
    else {
        target_speed = reference_speed;
        
        // SCC 모드가 아닐 때는 0으로 발행 (PlotJuggler에서 깔끔하게 보임)
        std_msgs::msg::Float32 zero_msg;
        zero_msg.data = 0.0f;
        p_d_range_->publish(zero_msg);
        p_safe_distance_->publish(zero_msg);
    }
    
    //=================================================
    // Forward-Backward Speed Profile Smoothing 적용
    // - 곡률 기반 속도 제한 + 가/감속 물리적 제한 적용
    reference_speed = SmoothSpeedProfile(driving_path_points, target_speed, vehicle_state.velocity);

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
// - 경로 포인트를 그대로 사용하여 FrenetConverter 초기화
//============================================
void PlanningNode::UpdateFrenetConverter(const interface::Lane &path_points) {
    // 변경 감지
    bool same_size = frenet_converter_initialized_ && (cached_path_points_.point.size() == path_points.point.size());
    const double eps = 1e-6;
    if (same_size) {
        for (size_t i = 0; i < path_points.point.size(); ++i) {
            double dx = std::abs(cached_path_points_.point[i].x - path_points.point[i].x);
            double dy = std::abs(cached_path_points_.point[i].y - path_points.point[i].y);
            if (dx > eps || dy > eps) {
                same_size = false;
                break;
            }
        }
    }
    if (same_size && frenet_converter_initialized_) {
        return;
    }

    if (path_points.point.size() < 2) {
        frenet_converter_initialized_ = false;
        return;
    }

    std::vector<double> x_pts;
    std::vector<double> y_pts;
    std::vector<double> psi_pts;
    x_pts.reserve(path_points.point.size());
    y_pts.reserve(path_points.point.size());
    psi_pts.reserve(path_points.point.size());

    for (size_t i = 0; i < path_points.point.size(); ++i) {
        x_pts.push_back(path_points.point[i].x);
        y_pts.push_back(path_points.point[i].y);

        if (i + 1 < path_points.point.size()) {
            double dx = path_points.point[i + 1].x - path_points.point[i].x;
            double dy = path_points.point[i + 1].y - path_points.point[i].y;
            psi_pts.push_back(std::atan2(dy, dx));
        } else if (!psi_pts.empty()) {
            psi_pts.push_back(psi_pts.back());
        } else {
            psi_pts.push_back(0.0);
        }
    }

    frenet_converter_ = FrenetConverter(x_pts, y_pts, psi_pts, false);

    cached_path_points_ = path_points;
    frenet_converter_initialized_ = true;
}

PlanningNode::FrenetCoordinate PlanningNode::CartesianToFrenet(double x_rel, double y_rel) {
    FrenetCoordinate frenet;

    if (!frenet_converter_initialized_) {
        return frenet;
    }

    auto [s, d] = frenet_converter_.cartesian_to_frenet(x_rel, y_rel);
    frenet.s = s;
    frenet.d = d;

    return frenet;
}

//============================================
//FrenetToCartesian 함수 구현 (Frenet -> Ego 좌표계 변환)
//============================================
std::pair<double, double> PlanningNode::FrenetToCartesian(double s, double d) {
    if (!frenet_converter_initialized_) {
        return {0.0, 0.0};
    }

    return frenet_converter_.frenet_to_cartesian(s, d);
}
//============================================
// LaneChange 함수 구현 (Frenet 경계조건 기반 경로 포인트 생성)
//============================================
interface::Lane PlanningNode::LaneChange(const interface::VehicleState &vehicle_state, const interface::PolyfitLane &driving_way, const BehaviorContext &ctx) {
    const double LANE_WIDTH = 4.0;
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
            RCLCPP_WARN(this->get_logger(), "[LaneChange] No available lane! All lanes blocked.");
            is_lane_changing_ = false;  // 차선 변경 중단
            return SamplePathFromPolyfit(driving_way);
        }
        
        target_lane_id_ = target_lane_id;
        
        //=============================================================
        // Frenet 경계 조건 설정
        //=============================================================
        double d0 = CartesianToFrenet(0.0, 0.0).d;  // 시작 d (현재 차선 중앙) 0.0->
        double df = (target_lane_id - current_lane_id_) * LANE_WIDTH;  // 목표 d (+4 또는 -4)
        
        // 장애물까지 거리를 sf로 사용 (단, 최소 거리 보장)
        double obstacle_s = ctx.static_object_s;    // = frenet.s
        const double LANE_CHANGE_MARGIN = 2.0;  // 최소 차선 변경 거리
        double sf = obstacle_s - LANE_CHANGE_MARGIN;  // 장애물 2m 전에 완료

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
        // sf 이후에도 새 차선을 따라 경로 연장 (안정화)
        //=============================================================
        lane_change_path_global_.clear();

        const double ds = 0.5;  // 0.5m 간격으로 샘플링
        const double STABILIZATION_DISTANCE = 5.0;  // sf 이후 15m 더 연장
        double s_max = sf + STABILIZATION_DISTANCE;
        
        for (double s = 0.0; s <= s_max; s += ds) {
            double d;
            
            if (s <= sf) {
                // sf까지: 3차 다항식으로 차선 변경
                d = frenet_a0 + frenet_a1 * s + frenet_a2 * s * s + frenet_a3 * s * s * s;
            } else {
                // sf 이후: 목표 차선(df)을 따라 직진
                d = df;
            }
            
            // Frenet → Local (FrenetConverter 사용)
            auto [x_local, y_local] = FrenetToCartesian(s, d);
            
            // Local → Global
            auto [x_global, y_global] = LocalToGlobal(vehicle_state, x_local, y_local);
            
            lane_change_path_global_.push_back({x_global, y_global});
        }
        
        // 목표점 (sf, df) Global 좌표 저장 (빨간 구체용)
        auto [target_x_local, target_y_local] = FrenetToCartesian(sf, df);
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

    const double ROI_MAX = 17.0; // 16m 이내 점들만 사용

    for (const auto& [gx, gy] : lane_change_path_global_) {
        auto [lx, ly] = GlobalToLocal(vehicle_state, gx, gy);
        
        // 차량 앞쪽 점만 사용 (x > -1m)
        if (lx > 0.0 && lx < ROI_MAX) {
            x_local_points.push_back(lx);
            y_local_points.push_back(ly);
        }
    }
    
    // 빨간 구체 Local 좌표
    auto [target_x_local, target_y_local] = GlobalToLocal(vehicle_state, 
                                                           target_point_global_x_, 
                                                           target_point_global_y_);

    // 로컬 경로 포인트 생성
    interface::Lane path_local = BuildLaneFromLocalPoints(
        x_local_points, y_local_points, driving_way.frame_id, driving_way.id);
    if (path_local.point.empty()) {
        path_local = SamplePathFromPolyfit(driving_way);
    }
    
    //=================================================================
    // 3) 완료 조건: 목표점을 충분히 지나고 새 차선에 안착
    //=================================================================
    // 목표점을 충분히 지나갔는지 확인 (5m 뒤로)
    bool passed_target = (target_x_local < -5.0);
    
    // 원래 차선 기준으로 현재 차량의 Frenet d 좌표 계산
    bool near_target_lane = false;
    double current_d_from_base = 0.0;
    
    if (base_frenet_converter_initialized_) {
        // 원래 차선 경로 기준으로 차량 위치 계산
        auto [s_base, d_base] = base_frenet_converter_.cartesian_to_frenet(0.0, 0.0);
        current_d_from_base = d_base;
        
        // 목표 d: 차선 변경 방향 * LANE_WIDTH
        double target_d = (target_lane_id_ - current_lane_id_) * LANE_WIDTH;
        
        // 목표 차선 근처에 있는지 확인 (0.8m 이내)
        near_target_lane = std::abs(current_d_from_base - target_d) < 0.8;
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 500,
            "[LaneChange] Progress: target_x=%.1f m, d_from_base=%.2f m, target_d=%.2f m, diff=%.2f m",
            target_x_local, current_d_from_base, target_d, std::abs(current_d_from_base - target_d));
    }
    
    bool timeout = (lane_change_counter_ > 1000);  // 10초
    bool no_points = (x_local_points.size() < 4);
    
    // 완료 조건: (목표점 통과 AND 새 차선 근처) OR 타임아웃 OR 경로 없음
    bool lane_change_complete = (passed_target && near_target_lane) || timeout || no_points;
    
    if (lane_change_complete) {
        current_lane_id_ = target_lane_id_;
        is_lane_changing_ = false;
        lane_change_path_saved_ = false;
        lane_change_path_global_.clear();

        RCLCPP_INFO(this->get_logger(), 
            "[LaneChange] DONE! new_lane=%d (passed=%d, near_lane=%d, d=%.2f m, timeout=%d, no_points=%d)",
            current_lane_id_, passed_target, near_target_lane, current_d_from_base, timeout, no_points);
        
        return SamplePathFromPolyfit(driving_way);
    }
    
    //=================================================================
    // 5) 시각화: 빨간 구체 (목표점)
    //=================================================================
    // frame_id를 명시적으로 설정 (ego/body는 차량 기준 로컬 좌표계)
    std::string viz_frame_id = cfg_.vehicle_namespace.empty() ? "ego/body" : cfg_.vehicle_namespace + "/body";
    
    visualization_msgs::msg::Marker target_marker;
    target_marker.header.frame_id = viz_frame_id;
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
    // 6) 시각화: 실제 경로 (초록 선) + 경로점들 (파란 점)
    //=================================================================
    visualization_msgs::msg::MarkerArray path_markers;
    
    // 6-1) 실제 경로 (초록 선)
    visualization_msgs::msg::Marker poly_line;
    poly_line.header.frame_id = viz_frame_id;
    poly_line.header.stamp = this->now();
    poly_line.ns = "lane_change_path";
    poly_line.id = 101;
    poly_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    poly_line.action = visualization_msgs::msg::Marker::ADD;
    
    for (const auto &pt : path_local.point) {
        geometry_msgs::msg::Point p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = 0.15;
        poly_line.points.push_back(p);
    }
    poly_line.scale.x = 0.15;
    poly_line.color.g = 1.0;  // 초록색
    poly_line.color.a = 1.0;
    path_markers.markers.push_back(poly_line);
    
    // 6-2) Frenet 경로점들 (파란 점) - 참고용
    visualization_msgs::msg::Marker path_points;
    path_points.header.frame_id = viz_frame_id;
    path_points.header.stamp = this->now();
    path_points.ns = "lane_change_points";
    path_points.id = 102;
    path_points.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    path_points.action = visualization_msgs::msg::Marker::ADD;
    
    for (size_t i = 0; i < path_local.point.size(); i++) {
        geometry_msgs::msg::Point p;
        p.x = path_local.point[i].x;
        p.y = path_local.point[i].y;
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
    
    return path_local;
}
int main(int argc, char **argv) {
    std::string node_name = "planning_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlanningNode>(node_name));
    rclcpp::shutdown();
    return 0;
}
