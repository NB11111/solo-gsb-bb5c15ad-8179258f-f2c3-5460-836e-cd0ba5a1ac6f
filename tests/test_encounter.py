"""双人轨迹相遇分析的单元测试。

可直接运行：python3 tests/test_encounter.py
或使用 pytest：python3 -m pytest tests/
"""
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.encounter import analyze_encounters
from app.geo import EARTH_RADIUS_M, TrackPoint

T0 = datetime(2026, 5, 1, 8, 0, 0, tzinfo=timezone.utc)

LAT0, LON0 = 31.0, 121.0
_COS0 = math.cos(math.radians(LAT0))


def xy_to_latlon(x, y):
    """局部米制坐标 -> 经纬度（与 app/geo.py 的等距圆柱投影互逆）。"""
    lat = LAT0 + math.degrees(y / EARTH_RADIUS_M)
    lon = LON0 + math.degrees(x / (_COS0 * EARTH_RADIUS_M))
    return lat, lon


def pt(x, y, s):
    lat, lon = xy_to_latlon(x, y)
    return TrackPoint(lat=lat, lon=lon, ele=None, time=T0 + timedelta(seconds=s))


def walk(x0, v, times, y=0.0):
    """匀速向东行走点列：x = x0 + v*t。"""
    return [pt(x0 + v * t, y, t) for t in times]


def stationary(x, y, times):
    return [pt(x, y, t) for t in times]


def every(step, end):
    return list(range(0, end + 1, step))


def test_basic_encounter():
    """A 以 1 m/s 经过静止的 B：距离 ≤ 50 米的时段为 [0,100]，合并为一次相遇。"""
    a = [walk(0.0, 1.0, every(10, 200))]      # 0→200 m，每 10 s 一点
    b = [stationary(50.0, 0.0, [0, 200])]     # B 停在 x=50
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    e = res.encounters[0]
    assert e.start_time == T0
    assert (e.end_time - T0).total_seconds() == 100.0
    assert abs(e.duration_s - 100.0) < 1e-6
    assert e.min_distance_m < 0.01            # A 恰好经过 B


def test_short_encounter_filtered():
    """快速掠过：处于 50 米内仅 10 秒（< 60 秒），不列出。"""
    a = [walk(0.0, 10.0, every(10, 200))]     # 10 m/s，0→2000 m
    b = [stationary(50.0, 0.0, [0, 200])]
    res = analyze_encounters(a, b)
    assert res.encounters == []


def test_no_time_overlap():
    """时间不重叠：无相遇。"""
    a = [stationary(0.0, 0.0, [0, 200])]
    b = [stationary(0.0, 0.0, [400, 600])]
    res = analyze_encounters(a, b)
    assert res.encounters == []


def test_gap_not_bridged():
    """A 断档（相邻采样间隔 > 10 分钟）：两侧各 50 秒的接近不跨断档拼成长相遇。"""
    # A：0~50 s 在原点附近，断档，1000~1050 s 再回到原点附近
    a = [[pt(0, 0, 0), pt(1, 0, 50), pt(2, 0, 1000), pt(3, 0, 1050)]]
    # B：停在原点，0~1050 s 连续覆盖（0→600 恰好 10 分钟，不算断档）
    b = [stationary(0.0, 0.0, [0, 600, 1050])]
    res = analyze_encounters(a, b)
    # 若错误地跨断档插值，会得到一次长相遇；正确口径下两段各 50 秒，均不足 60 秒
    assert res.encounters == []


def test_trkseg_boundary_not_bridged():
    """断在两个 trkseg：不跨 trkseg 虚构相遇，两段各 50 秒均不列出。"""
    a = [
        [pt(0, 0, 0), pt(1, 0, 50)],
        [pt(2, 0, 1000), pt(3, 0, 1050)],
    ]
    b = [stationary(0.0, 0.0, [0, 600, 1050])]
    res = analyze_encounters(a, b)
    assert res.encounters == []


def test_exactly_sixty_seconds_counts():
    """持续恰好 60 秒：列出（持续不足 60 秒才丢弃）。"""
    a = [walk(0.0, 1.0, every(10, 120))]
    b = [stationary(10.0, 0.0, [0, 120])]    # |t-10| ≤ 50 → t ∈ [0,60]
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    assert abs(res.encounters[0].duration_s - 60.0) < 0.01


def test_boundary_distance_inclusive():
    """距离恰好 50 米的时刻包含在相遇区间内（不超过 50 米）。"""
    # A 1 m/s，B 在 x=50：区间 [0,100]，端点距离恰好 50 米
    a = [walk(0.0, 1.0, every(10, 200))]
    b = [stationary(50.0, 0.0, [0, 200])]
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    assert (res.encounters[0].end_time - T0).total_seconds() == 100.0


def test_both_moving_min_distance_zero():
    """两人都在动：B 以 1 m/s 纵向穿越静止的 A，相遇 100 秒，最近距离 0。"""
    a = [stationary(0.0, 0.0, every(10, 120))]
    b = [[pt(0.0, t - 60.0, t) for t in every(10, 120)]]  # y: -60→60
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    e = res.encounters[0]
    assert abs(e.duration_s - 100.0) < 1e-6
    assert e.min_distance_m < 0.01


def test_closest_approach_misses():
    """最近距离 60 米（> 50 米）：二次不等式无实根区间，无相遇。"""
    a = [stationary(0.0, 0.0, every(10, 120))]
    b = [[pt(60.0, t - 60.0, t) for t in every(10, 120)]]
    res = analyze_encounters(a, b)
    assert res.encounters == []


def test_offset_passby_min_distance():
    """带 40 米横向偏移的穿越：相遇恰好 60 秒，最近距离 40 米。"""
    a = [stationary(0.0, 0.0, every(10, 120))]
    b = [[pt(40.0, t - 30.0, t) for t in every(10, 120)]]
    # 40² + (t-30)² ≤ 50² → |t-30| ≤ 30 → [0,60]
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    e = res.encounters[0]
    assert abs(e.duration_s - 60.0) < 0.01
    assert abs(e.min_distance_m - 40.0) < 0.05


def test_both_stationary_near():
    """两人都静止且相距 30 米：整个重叠时长 120 秒为一次相遇，最近距离 30 米。"""
    a = [stationary(0.0, 0.0, [0, 120])]
    b = [stationary(30.0, 0.0, [0, 120])]
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    e = res.encounters[0]
    assert abs(e.duration_s - 120.0) < 1e-6
    assert abs(e.min_distance_m - 30.0) < 0.05


def test_gap_exactly_ten_minutes_valid():
    """相邻采样间隔恰好 10 分钟（600 秒）：不视为断档，线段有效。"""
    a = [[pt(0, 0, 0), pt(0, 0, 600)]]
    b = [stationary(0.0, 0.0, [0, 600])]
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    assert abs(res.encounters[0].duration_s - 600.0) < 1e-6


def test_multiple_encounters_sorted():
    """断档前后两次相遇分别列出，按开始时间排序。"""
    a = [
        [pt(0, 0, 0), pt(1, 0, 100)],
        [pt(2, 0, 1000), pt(3, 0, 1100)],
    ]
    b = [stationary(0.0, 0.0, [0, 500, 1000, 1100])]
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 2
    e1, e2 = res.encounters
    assert (e1.start_time - T0).total_seconds() == 0.0
    assert abs(e1.duration_s - 100.0) < 1e-6
    assert (e2.start_time - T0).total_seconds() == 1000.0
    assert abs(e2.duration_s - 100.0) < 1e-6


def test_time_not_increasing_track1():
    a = [[pt(0, 0, 10), pt(0, 0, 5)]]
    b = [stationary(0.0, 0.0, [0, 120])]
    try:
        analyze_encounters(a, b)
    except ValueError as exc:
        assert "未递增" in str(exc) and "第 1 条轨迹" in str(exc)
    else:
        raise AssertionError("第 1 条轨迹时间倒序应报明确错误")


def test_time_not_increasing_track2():
    a = [stationary(0.0, 0.0, [0, 120])]
    b = [[pt(0, 0, 0), pt(0, 0, 0), pt(0, 0, 10)]]  # 零时差
    try:
        analyze_encounters(a, b)
    except ValueError as exc:
        assert "未递增" in str(exc) and "第 2 条轨迹" in str(exc)
    else:
        raise AssertionError("第 2 条轨迹零时差应报明确错误")


def test_single_point_segments_no_encounter():
    """只有单点 trkseg：无带时间线段可比较，返回空结果。"""
    a = [[pt(0, 0, 0)], [pt(10, 0, 60)]]
    b = [stationary(0.0, 0.0, [0, 120])]
    res = analyze_encounters(a, b)
    assert res.encounters == []


def test_intervals_merge_across_sample_boundaries():
    """同一连续覆盖时段内，跨越多个采样线段的相遇区间相接合并为一次。"""
    # 与 test_basic 相同构造，但显式断言只有一次（10 条采样线段产生的区间相接）
    a = [walk(0.0, 1.0, every(10, 200))]
    b = [stationary(50.0, 0.0, every(30, 200))]
    res = analyze_encounters(a, b)
    assert len(res.encounters) == 1
    assert abs(res.encounters[0].duration_s - 100.0) < 1e-6


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
