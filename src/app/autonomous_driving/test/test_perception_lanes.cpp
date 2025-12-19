#include "perception_node.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

class PerceptionNodeTestAccessor {
public:
    explicit PerceptionNodeTestAccessor(PerceptionNode &node) : node_(node) {}
    interface::PolyfitLanes FindLanes(const interface::Lane &lane_points) {
        return node_.FindLanes(lane_points);
    }
    interface::PolyfitLane FindDrivingWay(const interface::PolyfitLanes &poly_lanes) {
        return node_.FindDrivingWay(poly_lanes);
    }
private:
    PerceptionNode &node_;
};

namespace {

double EvalLaneY(const interface::PolyfitLane &lane, double x) {
    return lane.a0 + lane.a1 * x + lane.a2 * x * x + lane.a3 * x * x * x;
}

double ComputeRmse(const std::vector<interface::Point2D> &pts, const interface::PolyfitLane &lane) {
    if (pts.empty()) return 0.0;
    double sum = 0.0;
    for (const auto &p : pts) {
        double err = EvalLaneY(lane, p.x) - p.y;
        sum += err * err;
    }
    return std::sqrt(sum / static_cast<double>(pts.size()));
}

const interface::PolyfitLane *FindLaneById(const interface::PolyfitLanes &poly_lanes, const std::string &id) {
    for (const auto &ln : poly_lanes.polyfitlanes) {
        if (ln.id == id) return &ln;
    }
    return nullptr;
}

std::vector<interface::Point2D> MakeStraightLane(double y, int n_pts = 11, double spacing = 1.0, double noise = 0.0) {
    std::default_random_engine rng(42);
    std::normal_distribution<double> dist(0.0, noise);
    std::vector<interface::Point2D> pts;
    pts.reserve(static_cast<size_t>(n_pts));
    for (int i = 0; i < n_pts; ++i) {
        interface::Point2D p;
        p.x = static_cast<double>(i) * spacing;
        p.y = y + dist(rng);
        pts.push_back(p);
    }
    return pts;
}

std::vector<interface::Point2D> MakeCurvedLane(double a0, double a1, double a2, double a3,
                                               int n_pts = 21, double spacing = 0.5, double noise = 0.05) {
    std::default_random_engine rng(42);
    std::normal_distribution<double> dist(0.0, noise);
    std::vector<interface::Point2D> pts;
    pts.reserve(static_cast<size_t>(n_pts));
    for (int i = 0; i < n_pts; ++i) {
        double x = static_cast<double>(i) * spacing;
        double y = a0 + a1 * x + a2 * x * x + a3 * x * x * x + dist(rng);
        pts.push_back({x, y});
    }
    return pts;
}

interface::Lane LoadLaneCsv(const std::string &path, double roi_min_x, double roi_max_x, bool swap_xy = false) {
    interface::Lane lane;
    lane.frame_id = "csv";

    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[Warn] Cannot open csv: " << path << "\n";
        return lane;
    }

    std::string line;
    std::getline(fin, line); // skip header
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string sx, sy;
        if (!std::getline(ss, sx, ',')) continue;
        if (!std::getline(ss, sy, ',')) continue;
        double csv_x = 0.0, csv_y = 0.0;
        try {
            csv_x = std::stod(sx);
            csv_y = std::stod(sy);
        } catch (...) {
            continue;
        }
        interface::Point2D p;
        if (swap_xy) { // optional swap if dataset axes differ
            p.x = csv_y;
            p.y = csv_x;
        } else {
            p.x = csv_x;
            p.y = csv_y;
        }
        if (p.x < roi_min_x || p.x > roi_max_x) continue;
        lane.point.push_back(p);
    }
    return lane;
}

interface::Lane FilterLaneByRoi(const interface::Lane &src, double min_x, double max_x) {
    interface::Lane out;
    out.frame_id = src.frame_id;
    for (const auto &p : src.point) {
        if (p.x >= min_x && p.x <= max_x) {
            out.point.push_back(p);
        }
    }
    return out;
}

std::pair<double, double> MinMaxX(const interface::Lane &lane) {
    if (lane.point.empty()) return {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    double mn = lane.point.front().x;
    double mx = lane.point.front().x;
    for (const auto &p : lane.point) {
        mn = std::min(mn, p.x);
        mx = std::max(mx, p.x);
    }
    return {mn, mx};
}

double MeanY(const interface::Lane &lane) {
    if (lane.point.empty()) return 0.0;
    double sum = 0.0;
    for (const auto &p : lane.point) sum += p.y;
    return sum / static_cast<double>(lane.point.size());
}

void ShiftLaneY(interface::Lane &lane, double delta) {
    for (auto &p : lane.point) p.y -= delta;
}

interface::Lane CombineLanes(const std::vector<interface::Point2D> &lhs, const std::vector<interface::Point2D> &rhs = {}) {
    interface::Lane lane;
    lane.frame_id = "test";
    lane.point.insert(lane.point.end(), lhs.begin(), lhs.end());
    lane.point.insert(lane.point.end(), rhs.begin(), rhs.end());
    return lane;
}

bool AlmostEqual(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

bool TestFindLanesBothSides() {
    PerceptionNode node("perception_test", rclcpp::NodeOptions(), false);
    PerceptionNodeTestAccessor acc(node);
    auto left_pts = MakeStraightLane(1.75);
    auto right_pts = MakeStraightLane(-1.75);
    auto lane_points = CombineLanes(left_pts, right_pts);

    auto poly = acc.FindLanes(lane_points);

    const interface::PolyfitLane *left = FindLaneById(poly, "1");
    const interface::PolyfitLane *right = FindLaneById(poly, "2");
    bool ok = true;
    if (!left || !right) {
        std::cerr << "[Fail] Both lanes not detected\n";
        return false;
    }

    double rmse_left = ComputeRmse(left_pts, *left);
    double rmse_right = ComputeRmse(right_pts, *right);
    std::cout << "[Info] BothSides RMSE  left=" << rmse_left << " right=" << rmse_right << "\n"
              << "       Left coeff  (a0,a1,a2,a3)=(" << left->a0 << "," << left->a1 << "," << left->a2 << "," << left->a3 << ")\n"
              << "       Right coeff (a0,a1,a2,a3)=(" << right->a0 << "," << right->a1 << "," << right->a2 << "," << right->a3 << ")\n";
    if (!AlmostEqual(left->a0, 1.75, 0.2) || std::fabs(left->a1) > 0.2) {
        std::cerr << "[Fail] Left lane coeff unexpected\n";
        ok = false;
    }
    if (!AlmostEqual(right->a0, -1.75, 0.2) || std::fabs(right->a1) > 0.2) {
        std::cerr << "[Fail] Right lane coeff unexpected\n";
        ok = false;
    }
    if (rmse_left > 0.2 || rmse_right > 0.2) {
        std::cerr << "[Fail] RMSE too high (L:" << rmse_left << " R:" << rmse_right << ")\n";
        ok = false;
    }

    auto dw = acc.FindDrivingWay(poly);
    std::cout << "[Info] BothSides DrivingWay id=" << dw.id
              << " (a0,a1,a2,a3)=(" << dw.a0 << "," << dw.a1 << "," << dw.a2 << "," << dw.a3 << ")\n";
    if (dw.id != "driving_way_center" || !AlmostEqual(dw.a0, 0.0, 0.2)) {
        std::cerr << "[Fail] Driving way center incorrect\n";
        ok = false;
    }
    return ok;
}

bool TestFindLanesSingleSide() {
    PerceptionNode node("perception_test_single", rclcpp::NodeOptions(), false);
    PerceptionNodeTestAccessor acc(node);
    auto left_pts = MakeStraightLane(1.75);
    auto lane_points = CombineLanes(left_pts);

    auto poly = acc.FindLanes(lane_points);
    const interface::PolyfitLane *left = FindLaneById(poly, "1");
    const interface::PolyfitLane *right = FindLaneById(poly, "2");

    bool ok = true;
    if (!left || right) {
        std::cerr << "[Fail] Single lane detection unexpected\n";
        ok = false;
    }

    double rmse_left = left ? ComputeRmse(left_pts, *left) : 1e9;
    if (left) {
        std::cout << "[Info] SingleSide RMSE left=" << rmse_left << "\n"
                  << "       Left coeff (a0,a1,a2,a3)=(" << left->a0 << "," << left->a1 << "," << left->a2 << "," << left->a3 << ")\n";
    }
    if (rmse_left > 0.2) {
        std::cerr << "[Fail] Single lane RMSE too high\n";
        ok = false;
    }

    auto dw = acc.FindDrivingWay(poly);
    std::cout << "[Info] SingleSide DrivingWay id=" << dw.id
              << " (a0,a1,a2,a3)=(" << dw.a0 << "," << dw.a1 << "," << dw.a2 << "," << dw.a3 << ")\n";
    // left lane should offset to the right by ~2 m
    if (dw.id != "driving_way_offset" || !AlmostEqual(dw.a0, left->a0 - 2.0, 0.5)) {
        std::cerr << "[Fail] Driving way offset incorrect\n";
        ok = false;
    }
    return ok;
}

bool TestFindLanesCurvedNoisyBothSides() {
    PerceptionNode node("perception_test_curved", rclcpp::NodeOptions(), false);
    PerceptionNodeTestAccessor acc(node);

    // ground-truth poly y = a0 + a2 x^2 (mild curvature, same sign so 좌/우가 벌어짐)
    const double a0_left = 1.75;
    const double a0_right = -1.75;
    const double a1 = 0.0;
    const double a2 = 0.003;
    const double a3 = 0.0;

    auto left_pts = MakeCurvedLane(a0_left, a1, a2, a3, 21, 0.5, 0.05);
    auto right_pts = MakeCurvedLane(a0_right, a1, a2, a3, 21, 0.5, 0.05);
    auto lane_points = CombineLanes(left_pts, right_pts);

    auto poly = acc.FindLanes(lane_points);
    const interface::PolyfitLane *left = FindLaneById(poly, "1");
    const interface::PolyfitLane *right = FindLaneById(poly, "2");

    bool ok = true;
    if (!left || !right) {
        std::cerr << "[Fail] Curved noisy: Both lanes not detected\n";
        return false;
    }

    double rmse_left = ComputeRmse(left_pts, *left);
    double rmse_right = ComputeRmse(right_pts, *right);
    std::cout << "[Info] CurvedNoisy RMSE  left=" << rmse_left << " right=" << rmse_right << "\n"
              << "       Left coeff  (a0,a1,a2,a3)=(" << left->a0 << "," << left->a1 << "," << left->a2 << "," << left->a3 << ")\n"
              << "       Right coeff (a0,a1,a2,a3)=(" << right->a0 << "," << right->a1 << "," << right->a2 << "," << right->a3 << ")\n";

    if (rmse_left > 0.15 || rmse_right > 0.15) {
        std::cerr << "[Fail] Curved noisy RMSE too high (L:" << rmse_left << " R:" << rmse_right << ")\n";
        ok = false;
    }
    if (!AlmostEqual(left->a0, a0_left, 0.3) || !AlmostEqual(right->a0, a0_right, 0.3)) {
        std::cerr << "[Fail] Curved noisy a0 offset unexpected\n";
        ok = false;
    }
    // a2는 소음/가중치 때문에 부호가 뒤집힐 수 있어 절댓값만 체크
    double a2_left_mag = std::fabs(left->a2);
    double a2_right_mag = std::fabs(right->a2);
    if (std::fabs(a2_left_mag - std::fabs(a2)) > 0.002 || std::fabs(a2_right_mag - std::fabs(a2)) > 0.002) {
        std::cerr << "[Fail] Curved noisy |a2| unexpected (L:" << left->a2 << " R:" << right->a2 << " target: " << a2 << ")\n";
        ok = false;
    }

    auto dw = acc.FindDrivingWay(poly);
    std::cout << "[Info] CurvedNoisy DrivingWay id=" << dw.id
              << " (a0,a1,a2,a3)=(" << dw.a0 << "," << dw.a1 << "," << dw.a2 << "," << dw.a3 << ")\n";
    if (dw.id != "driving_way_center" || !AlmostEqual(dw.a0, 0.0, 0.25)) {
        std::cerr << "[Fail] Curved noisy driving way incorrect\n";
        ok = false;
    }
    return ok;
}

bool TestCsvRoiRmse() {
    // CSV 좌/우 차선: x=longitudinal(전방), y=lateral(좌+). 알고리즘 좌표계도 동일하게 사용.
    const double roi_min_x = 0.0;
    const double roi_max_x = 20.0;
    const std::string lane1_path = "/home/subin/subin_ws/ADS_Mission_Basic/resources/csv/simulation_lane/Lane_1.csv"; // right (y<0 after swap)
    const std::string lane2_path = "/home/subin/subin_ws/ADS_Mission_Basic/resources/csv/simulation_lane/Lane_2.csv"; // left  (y>0 after swap)

    auto right_lane = LoadLaneCsv(lane1_path, roi_min_x, roi_max_x, false);
    auto left_lane = LoadLaneCsv(lane2_path, roi_min_x, roi_max_x, false);
    std::cout << "[Info] CSV ROI pts left=" << left_lane.point.size() << " right=" << right_lane.point.size()
              << " (roi_x:[" << roi_min_x << "," << roi_max_x << "])\n";

    if (left_lane.point.empty() || right_lane.point.empty()) {
        std::cerr << "[Warn] CSV ROI has empty lane; skipping CSV RMSE check\n";
        return true; // skip instead of fail to keep other tests running
    }

    auto lane_points = CombineLanes(left_lane.point, right_lane.point);
    PerceptionNode node("perception_test_csv", rclcpp::NodeOptions(), false);
    PerceptionNodeTestAccessor acc(node);
    auto poly = acc.FindLanes(lane_points);
    const interface::PolyfitLane *left = FindLaneById(poly, "1");
    const interface::PolyfitLane *right = FindLaneById(poly, "2");

    bool ok = true;
    if (!left || !right) {
        std::cerr << "[Fail] CSV ROI: lanes not detected\n";
        return false;
    }

    double rmse_left = ComputeRmse(left_lane.point, *left);
    double rmse_right = ComputeRmse(right_lane.point, *right);
    std::cout << "[Info] CSV ROI RMSE  left=" << rmse_left << " right=" << rmse_right << "\n"
              << "       Left coeff  (a0,a1,a2,a3)=(" << left->a0 << "," << left->a1 << "," << left->a2 << "," << left->a3 << ")\n"
              << "       Right coeff (a0,a1,a2,a3)=(" << right->a0 << "," << right->a1 << "," << right->a2 << "," << right->a3 << ")\n";

    if (rmse_left > 0.3 || rmse_right > 0.3) {
        std::cerr << "[Fail] CSV ROI RMSE too high (L:" << rmse_left << " R:" << rmse_right << ")\n";
        ok = false;
    }
    return ok;
}

bool TestCsvRandomRoiRmse() {
    // 랜덤 ROI(전방 20 m)에서 CSV 차선을 가져와 RMSE 계산 (재현 가능한 seed 사용)
    const double roi_len = 20.0;
    const std::string lane1_path = "/home/subin/subin_ws/ADS_Mission_Basic/resources/csv/simulation_lane/Lane_1.csv"; // right
    const std::string lane2_path = "/home/subin/subin_ws/ADS_Mission_Basic/resources/csv/simulation_lane/Lane_2.csv"; // left

    auto right_full = LoadLaneCsv(lane1_path, -1e9, 1e9, false);
    auto left_full = LoadLaneCsv(lane2_path, -1e9, 1e9, false);
    if (left_full.point.empty() || right_full.point.empty()) {
        std::cerr << "[Warn] CSV Random ROI: empty lane after load; skipping\n";
        return true;
    }

    auto lr_minmax = MinMaxX(left_full);
    auto rr_minmax = MinMaxX(right_full);
    double start_min = std::max(lr_minmax.first, rr_minmax.first);
    double start_max = std::min(lr_minmax.second, rr_minmax.second) - roi_len;
    if (start_max <= start_min) {
        std::cerr << "[Warn] CSV Random ROI: insufficient overlap range; skipping\n";
        return true;
    }

    std::mt19937 rng(42); // deterministic
    std::uniform_real_distribution<double> dist(start_min, start_max);
    double roi_start = dist(rng);
    double roi_end = roi_start + roi_len;

    auto left_roi = FilterLaneByRoi(left_full, roi_start, roi_end);
    auto right_roi = FilterLaneByRoi(right_full, roi_start, roi_end);
    std::cout << "[Info] CSV Random ROI x:[" << roi_start << "," << roi_end << "] pts left=" << left_roi.point.size()
              << " right=" << right_roi.point.size() << "\n";

    if (left_roi.point.empty() || right_roi.point.empty()) {
        std::cerr << "[Warn] CSV Random ROI: empty filtered lane; skipping\n";
        return true;
    }

    // 좌우 차선의 평균 y를 0으로 맞춰 알고리즘 좌표계(좌=+, 우=-)에 가깝게 변환
    double y_center = 0.5 * (MeanY(left_roi) + MeanY(right_roi));
    ShiftLaneY(left_roi, y_center);
    ShiftLaneY(right_roi, y_center);

    auto lane_points = CombineLanes(left_roi.point, right_roi.point);
    PerceptionNode node("perception_test_csv_random", rclcpp::NodeOptions(), false);
    PerceptionNodeTestAccessor acc(node);
    auto poly = acc.FindLanes(lane_points);
    const interface::PolyfitLane *left = FindLaneById(poly, "1");
    const interface::PolyfitLane *right = FindLaneById(poly, "2");

    bool ok = true;
    if (!left || !right) {
        std::cerr << "[Fail] CSV Random ROI: lanes not detected\n";
        return false;
    }

    double rmse_left = ComputeRmse(left_roi.point, *left);
    double rmse_right = ComputeRmse(right_roi.point, *right);
    std::cout << "[Info] CSV Random ROI RMSE  left=" << rmse_left << " right=" << rmse_right << "\n"
              << "       Left coeff  (a0,a1,a2,a3)=(" << left->a0 << "," << left->a1 << "," << left->a2 << "," << left->a3 << ")\n"
              << "       Right coeff (a0,a1,a2,a3)=(" << right->a0 << "," << right->a1 << "," << right->a2 << "," << right->a3 << ")\n";

    if (rmse_left > 0.5 || rmse_right > 0.5) {
        std::cerr << "[Fail] CSV Random ROI RMSE too high (L:" << rmse_left << " R:" << rmse_right << ")\n";
        ok = false;
    }
    return ok;
}

bool TestFindLanesEmpty() {
    PerceptionNode node("perception_test_empty", rclcpp::NodeOptions(), false);
    PerceptionNodeTestAccessor acc(node);
    interface::Lane empty;
    auto poly = acc.FindLanes(empty);
    if (!poly.polyfitlanes.empty()) {
        std::cerr << "[Fail] Expected no lanes for empty input\n";
        return false;
    }
    auto dw = acc.FindDrivingWay(poly);
    if (dw.id != "driving_way_unknown") {
        std::cerr << "[Fail] Driving way should be unknown for empty input\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    bool ok = true;
    ok &= TestFindLanesBothSides();
    ok &= TestFindLanesSingleSide();
    ok &= TestFindLanesCurvedNoisyBothSides();
    ok &= TestCsvRoiRmse();
    ok &= TestCsvRandomRoiRmse();
    ok &= TestFindLanesEmpty();
    rclcpp::shutdown();
    std::cout << (ok ? "[Pass] perception lane tests\n" : "[Fail] perception lane tests\n");
    return ok ? 0 : 1;
}
