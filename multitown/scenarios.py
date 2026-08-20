"""Deterministic CyberTown decision scenarios with executable oracles."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    seed: int
    prompt: str
    allowed_actions: tuple[str, ...]
    oracle_action: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_actions"] = list(self.allowed_actions)
        return data


FACILITIES = ["hospital", "water_plant", "shelter", "data_center", "transit_hub"]
INCIDENTS = ["tunnel_fire", "chemical_leak", "bridge_fault", "grid_overload", "flood_gate"]
SOURCES = ["substation", "transformer", "control_room"]
ROUTES = ["north_route", "river_route", "central_route", "east_route"]
ZONES = ["old_town", "riverside", "school_zone", "industrial_zone", "hill_district"]
TOOLS = ["primary_drone", "inspection_robot", "field_team"]


def _render(title: str, facts: list[str], rule: str, actions: list[str]) -> str:
    action_json = json.dumps(actions, ensure_ascii=False)
    fact_text = "\n".join(f"- {fact}" for fact in facts)
    return (
        f"任务：{title}\n"
        f"已验证事实：\n{fact_text}\n\n"
        f"确定性决策规则：{rule}\n"
        f"允许动作：{action_json}\n\n"
        "严格只返回一个 JSON 对象，不要使用 Markdown："
        '{"action":"允许动作之一","confidence":0到1之间的数,"brief_reason":"不超过40个汉字"}'
    )


def _resource_allocation(seed: int, index: int) -> Scenario:
    rng = random.Random(seed)
    names = rng.sample(FACILITIES, 3)
    available = rng.randint(2, 4)
    rows: list[dict[str, int | str]] = []
    for name in names:
        rows.append({
            "name": name,
            "required": rng.randint(1, 5),
            "criticality": rng.randint(1, 5),
            "people": rng.randint(20, 220),
            "deadline": rng.randint(5, 45),
        })
    if not any(int(row["required"]) <= available for row in rows):
        rows[0]["required"] = available
    feasible = [row for row in rows if int(row["required"]) <= available]
    for row in rows:
        row["score"] = int(row["criticality"]) * 1000 + int(row["people"]) * 3 - int(row["deadline"]) * 5
    winner = max(feasible, key=lambda row: (int(row["score"]), str(row["name"])))
    actions = [f"allocate_{name}" for name in names] + ["request_mutual_aid"]
    facts = [f"可用抢修组={available}"] + [
        f"{row['name']}: 需要{row['required']}组，关键度{row['criticality']}，受影响{row['people']}人，失效倒计时{row['deadline']}分钟"
        for row in rows
    ]
    rule = "只考虑所需抢修组不超过可用数量的设施，优先级=关键度×1000+受影响人数×3-倒计时×5，选择优先级最高者。"
    return Scenario(
        f"resource-{index:03d}", "resource_allocation", seed,
        _render("把有限抢修组分配给一个设施", facts, rule, actions),
        tuple(actions), f"allocate_{winner['name']}", {"available": available, "facilities": rows},
    )


def _incident_dispatch(seed: int, index: int) -> Scenario:
    rng = random.Random(seed)
    names = rng.sample(INCIDENTS, 3)
    rows: list[dict[str, int | str | bool]] = []
    for name in names:
        travel, work = rng.randint(2, 20), rng.randint(3, 22)
        deadline = rng.randint(8, 35)
        severity, escalation = rng.randint(1, 5), rng.randint(0, 5)
        finish = travel + work
        rows.append({
            "name": name, "travel": travel, "work": work, "deadline": deadline,
            "severity": severity, "escalation": escalation, "finish": finish,
            "feasible": finish <= deadline,
            "score": severity * 100 + escalation * 20 - finish,
        })
    feasible = [row for row in rows if bool(row["feasible"])]
    actions = [f"dispatch_{name}" for name in names] + ["request_mutual_aid"]
    oracle = "request_mutual_aid" if not feasible else f"dispatch_{max(feasible, key=lambda row: (int(row['score']), str(row['name'])))['name']}"
    facts = [
        f"{row['name']}: 路程{row['travel']}分钟，处置{row['work']}分钟，截止{row['deadline']}分钟，严重度{row['severity']}，升级速度{row['escalation']}"
        for row in rows
    ]
    rule = "完成时间=路程+处置；仅选择完成时间不超过截止时间的事件，得分=严重度×100+升级速度×20-完成时间；无可行事件则请求外援。"
    return Scenario(
        f"dispatch-{index:03d}", "incident_dispatch", seed,
        _render("派出唯一应急队伍", facts, rule, actions), tuple(actions), oracle, {"incidents": rows},
    )


def _evidence_fusion(seed: int, index: int) -> Scenario:
    rng = random.Random(seed)
    reports: list[dict[str, int | str]] = []
    for reporter in ["sensor_a", "sensor_b", "resident_report", "maintenance_log"]:
        reports.append({"reporter": reporter, "source": rng.choice(SOURCES), "reliability": rng.randint(2, 9)})
    scores = {source: 0 for source in SOURCES}
    for report in reports:
        scores[str(report["source"])] += int(report["reliability"])
    ordered = sorted(SOURCES, key=lambda source: (scores[source], source), reverse=True)
    top, second = ordered[0], ordered[1]
    threshold, min_gap = 12, 3
    decisive = scores[top] >= threshold and scores[top] - scores[second] >= min_gap
    oracle = f"isolate_{top}" if decisive else f"inspect_{top}"
    actions = [f"inspect_{source}" for source in SOURCES] + [f"isolate_{source}" for source in SOURCES]
    facts = [f"{r['reporter']} 指向 {r['source']}，可靠度={r['reliability']}" for r in reports]
    rule = "每个故障源的证据分=所有指向它的可靠度之和。最高分至少12且领先第二名至少3分时隔离最高分故障源，否则先检查最高分故障源。"
    return Scenario(
        f"evidence-{index:03d}", "evidence_fusion", seed,
        _render("融合相互冲突的停电报告", facts, rule, actions), tuple(actions), oracle,
        {"reports": reports, "scores": scores, "threshold": threshold, "min_gap": min_gap},
    )


def _dependency_recovery(seed: int, index: int) -> Scenario:
    rng = random.Random(seed)
    nodes = rng.sample(["power_bus", "network_switch", "control_api", "dispatch_console", "alarm_service"], 4)
    root_index = rng.randint(0, 2)
    statuses = []
    for idx, node in enumerate(nodes):
        if idx < root_index:
            status = "healthy"
        elif idx == root_index:
            status = "offline"
        else:
            status = "degraded"
        statuses.append({"node": node, "status": status})
    actions = [f"repair_{node}" for node in nodes] + [f"restart_{nodes[-1]}"]
    oracle = f"repair_{nodes[root_index]}"
    facts = [f"依赖链={' -> '.join(nodes)}"] + [f"{row['node']}={row['status']}" for row in statuses]
    rule = "箭头右侧服务依赖左侧服务；先修复依赖链中最靠左的 offline 节点，不能先重启仅仅 degraded 的下游服务。"
    return Scenario(
        f"dependency-{index:03d}", "dependency_recovery", seed,
        _render("恢复小镇控制系统依赖链", facts, rule, actions), tuple(actions), oracle,
        {"nodes": nodes, "statuses": statuses, "root_index": root_index},
    )


def _supply_route(seed: int, index: int) -> Scenario:
    rng = random.Random(seed)
    names = rng.sample(ROUTES, 3)
    min_capacity, max_risk, max_time = rng.randint(30, 70), rng.randint(3, 7), rng.randint(20, 50)
    rows: list[dict[str, int | str | bool]] = []
    for name in names:
        capacity, risk, travel, fuel = rng.randint(20, 100), rng.randint(1, 10), rng.randint(10, 65), rng.randint(5, 30)
        feasible = capacity >= min_capacity and risk <= max_risk and travel <= max_time
        rows.append({
            "name": name, "capacity": capacity, "risk": risk, "travel": travel, "fuel": fuel,
            "feasible": feasible, "score": 1000 - travel * 10 - risk * 15 - fuel,
        })
    feasible = [row for row in rows if bool(row["feasible"])]
    actions = [f"use_{name}" for name in names] + ["request_airlift"]
    oracle = "request_airlift" if not feasible else f"use_{max(feasible, key=lambda row: (int(row['score']), str(row['name'])))['name']}"
    facts = [f"最低运力={min_capacity}，最大风险={max_risk}，最长时间={max_time}分钟"] + [
        f"{row['name']}: 运力{row['capacity']}，风险{row['risk']}，时间{row['travel']}分钟，燃料{row['fuel']}"
        for row in rows
    ]
    rule = "先过滤不满足最低运力、最大风险或最长时间的路线；可行路线得分=1000-时间×10-风险×15-燃料，选择最高分；无可行路线则请求空运。"
    return Scenario(
        f"route-{index:03d}", "supply_route", seed,
        _render("为医院选择补给路线", facts, rule, actions), tuple(actions), oracle,
        {"requirements": {"min_capacity": min_capacity, "max_risk": max_risk, "max_time": max_time}, "routes": rows},
    )


def _fault_recovery(seed: int, index: int) -> Scenario:
    rng = random.Random(seed)
    tool = rng.choice(TOOLS)
    retries = rng.randint(0, 3)
    fallback_safe = rng.choice([True, False])
    irreversible = rng.choice([True, False])
    if retries < 2 and not irreversible:
        oracle = f"retry_{tool}"
    elif fallback_safe:
        oracle = "switch_safe_fallback"
    else:
        oracle = "escalate_human_operator"
    actions = [f"retry_{tool}", "switch_safe_fallback", "escalate_human_operator", "ignore_failure"]
    facts = [
        f"失败工具={tool}", f"已经重试={retries}次", f"安全备用路径={'有' if fallback_safe else '无'}",
        f"继续错误操作是否不可逆={'是' if irreversible else '否'}",
    ]
    rule = "未达到2次重试且操作可逆时重试；否则若有安全备用路径则切换备用；其余情况升级人工，绝不忽略失败。"
    return Scenario(
        f"recovery-{index:03d}", "fault_recovery", seed,
        _render("处理执行体工具故障", facts, rule, actions), tuple(actions), oracle,
        {"tool": tool, "retries": retries, "fallback_safe": fallback_safe, "irreversible": irreversible},
    )


GENERATORS: tuple[Callable[[int, int], Scenario], ...] = (
    _resource_allocation,
    _incident_dispatch,
    _evidence_fusion,
    _dependency_recovery,
    _supply_route,
    _fault_recovery,
)


def build_scenario_bank(base_seed: int, count: int) -> list[Scenario]:
    """Build a balanced, deterministic bank whose IDs are stable across systems."""
    scenarios: list[Scenario] = []
    for index in range(count):
        generator = GENERATORS[index % len(GENERATORS)]
        scenario_seed = base_seed * 1009 + index * 7919
        scenarios.append(generator(scenario_seed, index))
    return scenarios
