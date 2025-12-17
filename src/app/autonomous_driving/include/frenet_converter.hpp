/*
 * frenet_converter.hpp
 * Frenet 좌표계 변환을 위한 클래스
 * - PolyfitLane을 샘플링하여 (x, y) waypoint 생성
 * - Cartesian ↔ Frenet 좌표 변환
 */
#ifndef __FRENET_CONVERTER_HPP__
#define __FRENET_CONVERTER_HPP__
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

class FrenetConverter {
//============================================
// 밖에서 쓸 수 있는 함수 목록 
// 1) FrenetConverter 생성자
// 2) cartesian_to_frenet: Cartesian -> Frenet 변환
// 3) frenet_to_cartesian: Frenet -> Cartesian 변환
// 4) get_max_s: 최대 s 값 반환
// 5) is_valid: 경로 유효성 확인
// 6) size: waypoint 개수 반환
//============================================
public:
    /**
     * @brief 생성자
     * @param x x좌표 벡터
     * @param y y좌표 벡터
     * @param psi heading 벡터 (사용하지 않지만 호환성 유지)
     * @param closed_loop 폐루프 여부
     */
    FrenetConverter(const std::vector<double>& x,
                    const std::vector<double>& y,
                    const std::vector<double>& psi,
                    bool closed_loop = false);
    
    //기본 생성자
    FrenetConverter() = default;
    
    /**
     * @brief Cartesian 좌표를 Frenet 좌표로 변환
     * @param x Cartesian x 좌표
     * @param y Cartesian y 좌표
     * @return (s, d) Frenet 좌표
     */
    std::pair<double, double> cartesian_to_frenet(double x, double y) const;
    
    /**
     * @brief Frenet 좌표를 Cartesian 좌표로 변환
     * @param s Frenet s 좌표 (누적 거리)
     * @param d Frenet d 좌표 (lateral offset)
     * @return (x, y) Cartesian 좌표
     */
    std::pair<double, double> frenet_to_cartesian(double s, double d) const;
    
    //최대 s 값 반환
    double get_max_s() const { return max_s_; }
    
    //경로가 유효한지 확인
    bool is_valid() const { return x_.size() >= 2; }
    
    //Waypoint 개수 반환
    size_t size() const { return x_.size(); }

private:
    // 두 점 사이의 거리 계산
    static double distance(double x1, double y1, double x2, double y2) {
        return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }
    
    //s 값들을 계산 (누적 거리)
    void compute_s_values();
    
    /**
     * @brief 주어진 점에서 가장 가까운 세그먼트 찾기
     * @param x 쿼리 점 x
     * @param y 쿼리 점 y
     * @param t 세그먼트 내 보간 계수 (출력)
     * @param ref_x 참조점 x (출력)
     * @param ref_y 참조점 y (출력)
     * @param ref_heading 참조점 heading (출력)
     * @param out_sqdist 최소 거리 제곱 (출력)
     * @return 세그먼트 시작 인덱스
     */
    int find_closest_segment(double x, double y, double& t,
                             double& ref_x, double& ref_y,
                             double& ref_heading, double& out_sqdist) const;
    
    /**
     * @brief s 값으로 세그먼트 찾기
     * @param s s 좌표
     * @param t 세그먼트 내 보간 계수 (출력)
     * @return 세그먼트 시작 인덱스
     */
    int locate_segment_by_s(double s, double& t) const;

private:
    std::vector<double> x_;           // x 좌표들
    std::vector<double> y_;           // y 좌표들
    std::vector<double> psi_;         // heading 값들 (호환성용)
    std::vector<double> s_;           // 누적 거리 (arc length)
    bool closed_loop_ = false;        // 폐루프 여부
    double max_s_ = 0.0;              // 최대 s 값
    double close_seg_len_ = 0.0;      // 폐루프 시 마지막 세그먼트 길이
};

#endif // __FRENET_CONVERTER_HPP__