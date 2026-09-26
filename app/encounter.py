"""双人轨迹相遇分析：求两条含时间轨迹在时间重叠期间相距不超过 50 米的连续区间。

构造口径（两条轨迹各自独立处理，不跨 trkseg 拼合）：

- 对每个原始 trkseg 的相邻点构造带时间线段，线段内位置按时间线性插值；
- 同一 trkseg 内相邻点时间必须严格递增，否则抛出 ValueError；
- 相邻采样间隔超过 10 分钟视为记录断档：断档两侧不构造线段，
  比较不会跨过断档（也不会跨过 trkseg 边界）。

比较口径：

- 仅当两条轨迹各取一条线段且二者的时间区间有正长度重叠时才比较；
- 两条轨迹投影到同一个局部等距圆柱米制平面（与 geo.py 口径一致），
  线段在平面内对时间是线性的，因此重叠时段内两人相对位置为
  d(τ) = d0 + v·τ，|d(τ)|² ≤ 50² 是关于 τ 的二次不等式，解析求解；
- 求解得到的子区间先按时间合并相接区间（只在双方都有有效线段覆盖的
  时间范围内相接才会合并，因此不会跨过断档或 trkseg 虚构相遇），
  合并后持续不足 60 秒的不列出；每次相遇给出起止时间、时长与最近距离。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .geo import EARTH_RADIUS_M

MEET_DISTANCE_M = 50.0    # 相遇距离阈值（米）：两人平面距离不超过该值
MIN_MEET_S = 60.0         # 最短相遇时长（秒）：持续不足 60 秒的区间不列出
MAX_SAMPLE_GAP_S = 600.0  # 记录断档阈值（秒）：相邻采样间隔超过 10 分钟不构造线段

_EPS_S = 1e-9             # 时长浮点噪声阈值（秒）
_MIN_DUR_EPS_S = 1e-3     # 最短时长判定容差（秒）：吸收局部投影/求根的数值误差


@dataclass
class TimedSeg:
    """一条带时间的平面线段：[start, end] 内位置 = (x0, y0) + (vx, vy)·秒。"""

    start: datetime
    end: datetime
    x0: float
    y0: float
    vx: float  # 米/秒
    vy: float


@dataclass
class _RawInterval:
    """一对线段时间重叠区内解出的一个相距 ≤ 阈值的子区间。"""

    start: datetime
    end: datetime
    lo: datetime       # 该重叠区的起点（x0/y0 的时间基准）
    x0: float
    y0: float
    vx: float
    vy: float

    def min_distance_m(self) -> float:
        """该子区间内两人的最近距离（相对运动为匀速直线，取钳制后的极值点）。"""
        v2 = self.vx * self.vx + self.vy * self.vy
        if v2 == 0.0:
            return math.hypot(self.x0, self.y0)
        tau = -(self.x0 * self.vx + self.y0 * self.vy) / v2
        lo_s = (self.start - self.lo).total_seconds()
        hi_s = (self.end - self.lo).total_seconds()
        tau = max(lo_s, min(hi_s, tau))
        return math.hypot(self.x0 + self.vx * tau, self.y0 + self.vy * tau)


@dataclass
class Encounter:
    """一次相遇：合并相接区间后持续不少于 60 秒的连续时段。"""

    start_time: datetime
    end_time: datetime
    duration_s: float
    min_distance_m: float


@dataclass
class EncounterResult:
    """相遇分析结果；encounters 按开始时间先后排列。"""

    encounters: list = field(default_factory=list)  # list[Encounter]


def _check_times(label: str, seg_idx: int, seg: list) -> None:
    """校验 trkseg 内相邻时间严格递增；零时差或倒序时抛出 ValueError。"""
    for i, (a, b) in enumerate(zip(seg, seg[1:]), start=1):
        if b.time <= a.time:
            raise ValueError(
                f"第 {label} 条轨迹的第 {seg_idx} 个 trkseg 中第 {i}→{i + 1} "
                "个轨迹点的时间未递增（零时差或倒序），无法进行相遇分析"
            )


def _build_timed_segments(label: str, segments: list, lat0: float, lon0: float) -> list:
    """把一条轨迹的各 trkseg 相邻点展开为带时间平面线段；断档处不构造。"""
    cos0 = math.cos(math.radians(lat0))
    timed: list = []
    for seg_idx, seg in enumerate(segments, start=1):
        _check_times(label, seg_idx, seg)
        for a, b in zip(seg, seg[1:]):
            dt = (b.time - a.time).total_seconds()  # 已校验严格递增，dt > 0
            if dt > MAX_SAMPLE_GAP_S:
                continue  # 超过 10 分钟的采样间隔视为断档，两侧不连线
            ax = math.radians(a.lon - lon0) * cos0 * EARTH_RADIUS_M
            ay = math.radians(a.lat - lat0) * EARTH_RADIUS_M
            bx = math.radians(b.lon - lon0) * cos0 * EARTH_RADIUS_M
            by = math.radians(b.lat - lat0) * EARTH_RADIUS_M
            timed.append(TimedSeg(
                start=a.time,
                end=b.time,
                x0=ax,
                y0=ay,
                vx=(bx - ax) / dt,
                vy=(by - ay) / dt,
            ))
    timed.sort(key=lambda s: s.start)
    return timed


def _pair_intervals(sa: TimedSeg, sb: TimedSeg, radius_m: float) -> list:
    """求两条线段时间重叠区内相距 ≤ radius_m 的子区间（解析二次不等式）。"""
    lo = max(sa.start, sb.start)
    hi = min(sa.end, sb.end)
    dur = (hi - lo).total_seconds()
    if dur <= _EPS_S:
        return []  # 仅端点相接或不重叠：无正长度区间

    # 重叠区起点 lo 处两人在共同平面中的位置与相对速度
    ta = (lo - sa.start).total_seconds()
    tb = (lo - sb.start).total_seconds()
    d0x = (sa.x0 + sa.vx * ta) - (sb.x0 + sb.vx * tb)
    d0y = (sa.y0 + sa.vy * ta) - (sb.y0 + sb.vy * tb)
    dvx = sa.vx - sb.vx
    dvy = sa.vy - sb.vy

    # |d0 + v·τ|² = r²  ⇔  A·τ² + B·τ + C = 0，求 ≤ 0 的区间
    A = dvx * dvx + dvy * dvy
    B = 2.0 * (d0x * dvx + d0y * dvy)
    C = d0x * d0x + d0y * d0y - radius_m * radius_m

    bounds = []
    if A == 0.0:
        if C <= 0.0:
            bounds = [(0.0, dur)]  # 相对静止（或同速平行）且全程在阈值内
    else:
        disc = B * B - 4.0 * A * C
        if disc <= 0.0:
            return []  # 无实根（全程在外）或仅相切（单点，不构成区间）
        sq = math.sqrt(disc)
        r1 = (-B - sq) / (2.0 * A)
        r2 = (-B + sq) / (2.0 * A)
        r1 = max(0.0, min(dur, r1))
        r2 = max(0.0, min(dur, r2))
        if r2 - r1 > _EPS_S:
            bounds = [(r1, r2)]

    out = []
    for t0, t1 in bounds:
        out.append(_RawInterval(
            start=lo + timedelta(seconds=t0),
            end=lo + timedelta(seconds=t1),
            lo=lo,
            x0=d0x,
            y0=d0y,
            vx=dvx,
            vy=dvy,
        ))
    return out


def _merge_intervals(raws: list) -> list:
    """按时间合并相接区间。

    原始区间只可能位于两条轨迹都有有效线段覆盖的时间范围内，
    故相接合并只会连接连续覆盖的时段，不会跨过断档或 trkseg 空档。
    """
    raws.sort(key=lambda r: r.start)
    merged: list = []
    for raw in raws:
        if merged and raw.start <= merged[-1][-1].end + timedelta(seconds=_EPS_S):
            merged[-1].append(raw)
        else:
            merged.append([raw])
    return merged


def analyze_encounters(segments_a: list, segments_b: list) -> EncounterResult:
    """分析两条轨迹（各自为 list[list[TrackPoint]]）的相遇区间。

    任一条轨迹相邻点时间未严格递增时抛出 ValueError；
    无有效线段（断档/单点）或时间不重叠时返回空结果。
    """
    # 以两条轨迹首点的中点为共同局部平面原点；相遇时两人位置几乎相同，
    # 原点远近不影响近距离相对向量的精度
    lat0 = (segments_a[0][0].lat + segments_b[0][0].lat) / 2.0
    lon0 = (segments_a[0][0].lon + segments_b[0][0].lon) / 2.0
    segs_a = _build_timed_segments(1, segments_a, lat0, lon0)
    segs_b = _build_timed_segments(2, segments_b, lat0, lon0)

    raws: list = []
    i = j = 0
    while i < len(segs_a) and j < len(segs_b):
        sa, sb = segs_a[i], segs_b[j]
        if sa.end < sb.start:
            i += 1
            continue
        if sb.end < sa.start:
            j += 1
            continue
        raws.extend(_pair_intervals(sa, sb, MEET_DISTANCE_M))
        # 同一条轨迹内各线段时间互不重叠（断档/trkseg 之间有空档），
        # 结束较早者推进；恰好在同一时刻结束则两条都推进
        if sa.end < sb.end:
            i += 1
        elif sb.end < sa.end:
            j += 1
        else:
            i += 1
            j += 1

    encounters: list = []
    for group in _merge_intervals(raws):
        start = group[0].start
        end = max(r.end for r in group)
        duration = (end - start).total_seconds()
        if duration < MIN_MEET_S - _MIN_DUR_EPS_S:
            continue  # 持续不足 60 秒的相遇不列出（留数值容差吸收投影误差）
        min_dist = min(r.min_distance_m() for r in group)
        encounters.append(Encounter(
            start_time=start,
            end_time=end,
            duration_s=duration,
            min_distance_m=min_dist,
        ))

    encounters.sort(key=lambda e: e.start_time)
    return EncounterResult(encounters=encounters)
