/**
 * @copyright Hanyang University, Department of Automotive Engineering, 2024. All rights reserved. 
 *            Subject to limited distribution and restricted disclosure only.
 *            
 * @file      autonomous_driving_config.hpp
 * @brief     autonomous driving configuration
 * 
 * @date      2023-08-07 created by Yuseung Na (yuseungna@hanyang.ac.kr)
 */

#ifndef __AUTONOMOUS_DRIVING_CONFIG_HPP__
#define __AUTONOMOUS_DRIVING_CONFIG_HPP__
#pragma once

// STD Header
#include <string>
#include <cmath>

typedef struct {
    std::string vehicle_namespace{""};
    double loop_rate_hz{100.0};
    bool use_manual_inputs{true};

    ////////////////////// TODO //////////////////////
    // [다훈 수정 5]TODO: Add more parameters (ex: kd, ki, kv, ...) from autonomous_driving.hpp
    //1. 차량 물리 파라미터
    double param_wheel_base{1.302 + 1.398}; // L_f + L_r
    double param_max_lateral_accel{6200.0 / 1319.91}; // Fyf_max / Mass

    //2. Pure Pursuit 제어 파라미터
    double param_pp_kd{5.0};
    double param_pp_kv{0.2};
    double param_pp_kc{0.0};
    double param_pp_ice_lookahead_scale{1.5}; // 빙판길 시 look-ahead 확대
    double param_pp_ice_steer_scale{0.7};     // 빙판길 시 조향 완화 스케일

    //3. PID 제어 파라미터
    double param_pid_kp{5.0};
    double param_pid_ki{0.002};
    double param_pid_kd{0.0};
    double param_brake_ratio{1.2};

    //4. Speed Profile Smoothing 파라미터 (Forward-Backward)
    double param_a_max{2.0};   // 최대 가속도 [m/s²]
    double param_a_min{-3.0};  // 최대 감속도 [m/s²] (음수)
    double param_slope_ff_up{0.6};     // uphill feed-forward accel [m/s²]
    double param_slope_ff_down{0.6};   // downhill feed-forward brake [m/s²]

    //5. ROI 파라미터 (Perception에서 lane point 필터링용)
    // - Front/Rear: x축 방향 (앞쪽 +, 뒤쪽 -)
    // - Left/Right: y축 방향 (왼쪽 +, 오른쪽 -)
    double param_m_ROIFront_param{15.0};  // 앞쪽 ROI [m]
    double param_m_ROIRear_param{5.0};    // 뒤쪽 ROI [m]
    double param_m_ROILeft_param{3.0};    // 왼쪽 ROI [m]
    double param_m_ROIRight_param{3.0};   // 오른쪽 ROI [m]
    std::string ref_csv_path{""};

    // [다훈 수정9] Control Parameters (for longitudinal control)
    const double dt{1.0/100.0};
    
    //Algorithm Parameters
    double speed_error_integral{0.0};
    double speed_error_prev{0.0};

    //////////////////////////////////////////////////
} AutonomousDrivingConfig;

#endif // __AUTONOMOUS_DRIVING_CONFIG_HPP__
