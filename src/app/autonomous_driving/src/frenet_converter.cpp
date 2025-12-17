/*
 * frenet_converter.cpp
 * Frenet 좌표계 변환 구현
 */
#include "frenet_converter.hpp"

FrenetConverter::FrenetConverter(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 const std::vector<double>& psi,
                                 bool closed_loop)
: x_(x), y_(y), psi_(psi), closed_loop_(closed_loop) {
    compute_s_values();
}

//============================================
//1. compute_s_values()
//  x_pts, y_pts, psi_pts를 기반으로 누적 거리(s_[i]) 계산
//  open_loop :s 넘어가면 더 이상 길이 없음 
//  closed_loop: 마지막 점에서 첫 점으로 이어짐 (ex, F1tenth)
//============================================
void FrenetConverter::compute_s_values() {
    const size_t N = x_.size();                //N = 생성한 x_pts (waypoint) 개수
    s_.assign(N, 0.0); // 초기화                //s_[0] = 0.0 크기:N 배열
    if (N < 2) { max_s_ = 0.0; return; } 

    for (size_t i = 1; i < N; ++i) {
        s_[i] = s_[i-1] + distance(x_[i-1], y_[i-1], x_[i], y_[i]);
    }
    // 폐루프면 last->first 구간 포함
    if (closed_loop_) {
        close_seg_len_ = distance(x_.back(), y_.back(), x_.front(), y_.front());
        max_s_ = s_.back() + close_seg_len_;
    } else {
        close_seg_len_ = 0.0;
        max_s_ = s_.back();                 //시작점~s끝점까지 누적
    }
}

//============================================
//2. find_closest_segment()
//  cartesian_to_frenet()에서 사용
//  segment (두 waypoint를 잇는 선분 하나)
//  segment i = i번째 waypoint -> i+1번째 waypoint
//============================================
int FrenetConverter::find_closest_segment(double x, double y, double& t,
                                          double& ref_x, double& ref_y,
                                          double& ref_heading, double& out_sqdist) const {
    const size_t N = x_.size();
    int best_i = 0;                                             //지금까지 본 세그먼트들 중에서 (x,y)와 가장 가까운 세그먼트 시작 index
    double best_t = 0.0;                                        //세그먼트 내에서의 상대 위치(0~1)
    double best_sq = std::numeric_limits<double>::infinity();   //최소 거리 제곱
    double best_rx = 0.0, best_ry = 0.0, best_heading = 0.0;    //최소 거리일 때의 투영점 좌표와 heading

    const auto check_seg = [&](size_t i, size_t j) {            //lamda 함수: 한 개 세그먼트에 대해 거리 계산
        double vx = x_[j] - x_[i];                              //세그먼트 방향 벡터
        double vy = y_[j] - y_[i];
        double seg_len2 = vx * vx + vy * vy;                    //세그먼트 길이 제곱
        if (seg_len2 < 1e-12) return; // 너무 짧으면 무시

        //(x,y)를 세그먼트에 수선 투영
        // w = (wx, wy) = (x - x_i, y - y_i) 
        // 수선 투영 계수 t
        double wx = x - x_[i];                                  //세그먼트 시작점 i에서 부터 (x,y)까지의 벡터 
        double wy = y - y_[i];
        double tt = (wx * vx + wy * vy) / seg_len2;             //내적 공식 t_raw = w*v / |v|^2
        tt = std::max(0.0, std::min(1.0, tt));                  //세그먼트 내에 투영되도록 0~1로 클램프    t_raw < 0 뒤쪽에 (x,y)의 투영 점이 위치 t_raw > 1 앞쪽에

        double projx = x_[i] + tt * vx;                         //투영 계수 tt를 이용해 세그먼트 상의 투영점(projx, projy) 좌표 계산
        double projy = y_[i] + tt * vy;
        double dx = x - projx;                                  //(x,y)와 투영점과의 거리 벡터
        double dy = y - projy;
        double sq = dx * dx + dy * dy;                          //(x,y)와 투영점과의 거리 제곱

        if (sq < best_sq) {
            best_sq = sq;
            best_i = static_cast<int>(i);
            best_t = tt;
            best_rx = projx;
            best_ry = projy;
            best_heading = std::atan2(vy, vx); // 세그먼트 진행방향
        }
    };

    // 0..N-2 세그먼트
    for (size_t i = 0; i + 1 < N; ++i) check_seg(i, i + 1);     //모든 세그먼트에 대해 check_seg 수행
    // 폐루프면 N-1 -> 0 세그먼트도
    if (closed_loop_ && N >= 2) check_seg(N - 1, 0);

    t = best_t;
    ref_x = best_rx;
    ref_y = best_ry;
    ref_heading = best_heading;
    out_sqdist = best_sq;
    return best_i;
}

//============================================
//3. locate_segment_by_s()
//  frenet_to_cartesian()에서 사용
//============================================
int FrenetConverter::locate_segment_by_s(double s, double& t) const {
    // s를 [0, max_s_)로 래핑 -> 즉, s가 한바퀴 돌아도 다시 0~max_s_ 범위로 집어넣음
    if (max_s_ > 0.0) {
        while (s < 0.0) s += max_s_;
        while (s >= max_s_) s -= max_s_;
    }

    const size_t N = x_.size();
    if (!closed_loop_) {
        // 열린 경로: s_[i] <= s < s_[i+1] 를 찾음
        auto it = std::upper_bound(s_.begin(), s_.end(), s);
        size_t idx = (it == s_.begin()) ? 0 : (static_cast<size_t>(it - s_.begin()) - 1);
        if (idx >= N - 1) { idx = N - 2; t = 1.0; return static_cast<int>(idx); }
        double seg_len = s_[idx + 1] - s_[idx];
        t = (seg_len > 1e-9) ? ((s - s_[idx]) / seg_len) : 0.0;
        t = std::clamp(t, 0.0, 1.0);
        return static_cast<int>(idx);
    } else {
        // 폐루프: s가 마지막 포인트 이후면 마지막->첫점 세그먼트
        if (s >= s_.back()) {
            double seg_len = close_seg_len_;
            t = (seg_len > 1e-9) ? ((s - s_.back()) / seg_len) : 0.0;
            t = std::clamp(t, 0.0, 1.0);
            return static_cast<int>(N - 1); // N-1 -> 0
        } else {
            auto it = std::upper_bound(s_.begin(), s_.end(), s);
            size_t idx = (it == s_.begin()) ? 0 : (static_cast<size_t>(it - s_.begin()) - 1);
            if (idx >= N - 1) { idx = N - 2; t = 1.0; return static_cast<int>(idx); }
            double seg_len = s_[idx + 1] - s_[idx];
            t = (seg_len > 1e-9) ? ((s - s_[idx]) / seg_len) : 0.0;
            t = std::clamp(t, 0.0, 1.0);
            return static_cast<int>(idx);
        }
    }
}

std::pair<double, double> FrenetConverter::cartesian_to_frenet(double x, double y) const {
    if (x_.size() < 2) return {0.0, 0.0};
    
    double t, rx, ry, heading, sq;
    int i = find_closest_segment(x, y, t, rx, ry, heading, sq);

    // 세그먼트 기하
    size_t j = (static_cast<size_t>(i) + 1 < x_.size()) ? (i + 1) : 0;
    double vx = x_[j] - x_[i];
    double vy = y_[j] - y_[i];
    double seg_len = std::sqrt(vx * vx + vy * vy);
    if (seg_len < 1e-12) return {0.0, 0.0};

    // 진행방향 단위벡터/법선
    double tx = vx / seg_len, ty = vy / seg_len;
    double nx = -ty, ny = tx; // 왼쪽 양수                                  //(x,y)->(-y,x) 왼쪽으로 90도 회전

    // s값 계산 (누적 거리)                                                                                            // 조건 ? A: B -> 조건이 true이면 A, false이면 B
    double s = s_[i] + t * ((static_cast<size_t>(j) > static_cast<size_t>(i)) ? (s_[j] - s_[i]) : close_seg_len_); //s_[i]:compute_s_values()에서 계산한 누적 거리
    if (closed_loop_) {
        if (s >= max_s_) s -= max_s_;
        if (s < 0.0) s += max_s_;
    } else {
        s = std::clamp(s, 0.0, max_s_);
    }

    // d값 계산 d = (p - p_ref)·n̂
    double dx = x - rx, dy = y - ry;
    double d = dx * nx + dy * ny;

    return {s, d};
}

std::pair<double, double> FrenetConverter::frenet_to_cartesian(double s, double d) const {
    if (x_.size() < 2) return {0.0, 0.0};
    
    double t;
    int i = locate_segment_by_s(s, t);

    size_t N = x_.size();
    size_t j = (static_cast<size_t>(i) + 1 < N) ? (i + 1) : 0;

    // 세그먼트 보간 ref point
    double ref_x, ref_y, vx, vy;
    if (j != 0) {
        ref_x = x_[i] + t * (x_[j] - x_[i]);
        ref_y = y_[i] + t * (y_[j] - y_[i]);
        vx = x_[j] - x_[i];
        vy = y_[j] - y_[i];
    } else {
        // N-1 -> 0
        ref_x = x_[i] + t * (x_[0] - x_[i]);
        ref_y = y_[i] + t * (y_[0] - y_[i]);
        vx = x_[0] - x_[i];
        vy = y_[0] - y_[i];
    }
    double seg_len = std::sqrt(vx * vx + vy * vy);
    if (seg_len < 1e-12) return {ref_x, ref_y};

    double tx = vx / seg_len, ty = vy / seg_len;
    double nx = -ty, ny = tx; // 왼쪽 양수

    double X = ref_x + d * nx;
    double Y = ref_y + d * ny;
    return {X, Y};
}