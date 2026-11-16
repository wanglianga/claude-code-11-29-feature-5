"""Exclusive set packing over candidate events.

Objective, in strict lexicographic order:

1. maximise the number of *reliably explained* observations (weight 1 for
   observations whose event window has fully occurred on every reachable
   station; weight 0 for observations that could still change because a
   station has not reported that interval), then
2. maximise the total number of explained observations, then
3. minimise the total residual (sum of squared-residual equivalents).

Level 2 is the tie-break that keeps a run whose stations have not yet reported
progress from collapsing to one low-residual event: an observation explained
*somewhere* is better than one left unassigned even if nothing is finalisable
yet.

Constraints: a candidate is either taken whole or not at all; no observation
appears in two taken candidates.  Every candidate is a clique with at most one
record per station, so "at most one record per station per event" holds by
construction.

Exactness
---------
Candidates that share no observation cannot interact, so the problem is split
into the *connected components* of the conflict graph.  Each component is solved
to optimality with depth-first branch-and-bound, then the component optima are
combined: the lexicographic score is additive, so per-component optima give the
global optimum.  A component that exceeds the node budget is reported as
non-proven and its best-found solution is kept, and the run's notes say so.

Ties: when two distinct decompositions of the *same* component tie exactly on
all objectives, the component is ambiguous.  The run is then reported as
ambiguous and requires a human choice; no option is silently preferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .association import Candidate
from .config import Settings

KEEP_LIMIT = 100


@dataclass
class SolveSolution:
    option_id: str
    candidate_ids: list[str]
    explained_observations: int
    reliable_observations: int
    total_residual_sq_us2: int
    event_count: int
    observation_ids: list[int] = field(default_factory=list)


@dataclass
class ComponentSolution:
    candidates: list[Candidate]
    best_value: int
    top: list[tuple[int, list[int], int, int]]  # value, indices, explained, residual
    proven: bool
    nodes: int


@dataclass
class SolveResult:
    best: SolveSolution | None
    options: list[SolveSolution] = field(default_factory=list)
    tied: bool = False
    proven_optimal: bool = False
    nodes_explored: int = 0
    notes: list[str] = field(default_factory=list)
    components: int = 0


def _components(candidates: list[Candidate]) -> list[list[int]]:
    """Connected components of the "shares an observation" graph."""
    n = len(candidates)
    obs_sets = [frozenset(c.observation_ids) for c in candidates]
    owner: dict[int, list[int]] = {}
    for i, s in enumerate(obs_sets):
        for o in s:
            owner.setdefault(o, []).append(i)
    adj: list[set[int]] = [set() for _ in range(n)]
    for ids in owner.values():
        for a in ids:
            adj[a].update(x for x in ids if x != a)
    seen = [False] * n
    out: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        out.append(sorted(comp))
    return out


def _solve_component(
    cands: list[Candidate],
    weight_by_observation: dict[int, int],
    cfg: Settings,
    node_budget: int,
) -> ComponentSolution:
    """Solve one independent component, returning indices *into* ``cands``."""
    total_residual = sum(c.residual_sq_us2 for c in cands)
    # ``BIG_E`` exceeds any achievable total residual, so the residual term can
    # never borrow from the "explained" digit; ``BIG_R`` exceeds everything the
    # two lower levels can contribute, so one more reliably-explained record
    # always beats any combination of the lower levels.
    big_e = total_residual + 1
    max_explained = sum(len(c.observation_ids) for c in cands)
    big_r = (max_explained + 2) * big_e
    values = []
    for c in cands:
        reliable = sum(weight_by_observation.get(o, 0) for o in c.observation_ids)
        values.append(
            reliable * big_r + len(c.observation_ids) * big_e - c.residual_sq_us2
        )
    n = len(cands)
    obs_sets = [frozenset(c.observation_ids) for c in cands]
    # Upper bound on what the suffix can still contribute.  Chosen candidates are
    # pairwise disjoint, so no observation can be counted twice:
    #   * at most ``distinct`` observations, each weighted by its reliability;
    #   * at most (n - i) candidates, each contributing at most (big_e - 1) in
    #     its bonus term, since residuals are non-negative;
    #   * residual therefore contributes none of the subtraction.
    suffix = [0] * (n + 1)
    counts: dict[int, int] = {}
    distinct = 0
    reliable_sum = 0
    for i in range(n - 1, -1, -1):
        for o in obs_sets[i]:
            if counts.get(o, 0) == 0:
                distinct += 1
                reliable_sum += weight_by_observation.get(o, 0)
            counts[o] = counts.get(o, 0) + 1
        # Residual only ever subtracts, so the optimistic bound omits it.
        suffix[i] = reliable_sum * big_r + distinct * big_e

    kept: list[tuple[int, list[int], int, int]] = []
    nodes = 0
    exhausted = True
    # Incumbent: the best score seen so far.  The bound is strict, so a branch
    # that could only *tie* the incumbent is still explored and its tied
    # decomposition is kept for the human selection step.
    best_found = [0]

    def consider(chosen: list[int], score: int, explained: int, residual: int) -> None:
        if score < best_found[0]:
            # This branch is strictly worse than the incumbent; keeping it
            # would only crowd out genuinely tied alternatives.
            return
        if score > best_found[0]:
            best_found[0] = score
            kept.clear()
        kept.append((score, sorted(chosen), explained, residual))
        if len(kept) > KEEP_LIMIT:
            kept.sort(key=lambda e: (-e[0], e[1]))
            del kept[KEEP_LIMIT:]

    def rec(idx: int, chosen: list[int], score: int, explained: int, residual: int) -> None:
        nonlocal nodes, exhausted
        nodes += 1
        if nodes > node_budget:
            exhausted = False
            return
        if idx >= n:
            if chosen:
                consider(chosen, score, explained, residual)
            return
        if score + suffix[idx] < best_found[0]:
            return
        rec(idx + 1, chosen, score, explained, residual)
        if not exhausted:
            return
        for i in chosen:
            if obs_sets[i] & obs_sets[idx]:
                return
        rec(
            idx + 1,
            chosen + [idx],
            score + values[idx],
            explained + len(cands[idx].observation_ids),
            residual + cands[idx].residual_sq_us2,
        )

    rec(0, [], 0, 0, 0)
    best_value = kept[0][0] if kept else 0
    top = [e for e in kept if e[0] == best_value]
    return ComponentSolution(
        candidates=cands,
        best_value=best_value,
        top=top[: cfg.max_solution_options],
        proven=exhausted,
        nodes=nodes,
    )


def solve(
    candidates: list[Candidate],
    weight_by_observation: dict[int, int],
    cfg: Settings,
) -> SolveResult:
    if not candidates:
        return SolveResult(best=None, proven_optimal=True, notes=["no candidates"])

    # Global ordering: high value first, then low residual, then stable id.
    ordered = sorted(
        candidates,
        key=lambda c: (
            -sum(weight_by_observation.get(o, 0) for o in c.observation_ids),
            c.residual_sq_us2,
            c.candidate_id,
        ),
    )
    # Drop candidates with an identical observation set, keeping the lowest
    # residual one: two of them can never both be chosen.
    best_for_set: dict[frozenset[int], Candidate] = {}
    for c in ordered:
        key = frozenset(c.observation_ids)
        prev = best_for_set.get(key)
        if prev is None or c.residual_sq_us2 < prev.residual_sq_us2:
            best_for_set[key] = c
    # Identity, not equality: Candidate has no __eq__, but filtering by value
    # lookup would keep look-alike candidates from different components.
    keep_ids = {id(c) for c in best_for_set.values()}
    ordered = [c for c in ordered if id(c) in keep_ids]

    comps = _components(ordered)
    budget_each = max(1, cfg.max_solver_nodes // max(1, len(comps)))
    solutions: list[ComponentSolution] = []
    total_nodes = 0
    all_proven = True
    for comp in comps:
        sub = [ordered[i] for i in comp]
        sol = _solve_component(sub, weight_by_observation, cfg, budget_each)
        # Map local indices back to positions in the global ``ordered`` list.
        sol.top = [
            (value, [comp[j] for j in indices], explained, residual)
            for value, indices, explained, residual in sol.top
        ]
        solutions.append(sol)
        total_nodes += sol.nodes
        all_proven = all_proven and sol.proven

    notes = [
        "objective is lexicographic: maximise reliably explained observations, then "
        "total explained observations, then minimise total residual; a candidate is "
        "accepted or rejected whole and no observation is used by two events",
        f"conflict graph split into {len(comps)} independent component(s); each was "
        "solved separately, which is exact because components share no observation",
    ]
    if not all_proven:
        notes.append(
            f"solver stopped after {cfg.max_solver_nodes} nodes in at least one "
            "component; optimality is not proven for those components"
        )

    ambiguous_components = [i for i, s in enumerate(solutions) if len(s.top) > 1]
    tie = bool(ambiguous_components)

    # Enumerate the cross-product of the top decompositions per component,
    # keeping the worst component score fixed at its optimum (any other choice
    # is strictly worse).  This yields alternative global options that differ
    # only where components are genuinely ambiguous.
    primary: list[int] = []
    variants: list[list[int]] = []
    for sol in solutions:
        primary.extend(sol.top[0][1] if sol.top else [])
        if len(sol.top) > 1:
            variants.append(sol.top)
    options_entries: list[dict] = []
    options_entries.append({"indices": primary, "source": "component optima"})
    for comp_index in ambiguous_components:
        for alt in solutions[comp_index].top[1:]:
            variant = []
            for i, sol in enumerate(solutions):
                pick = alt[1] if i == comp_index else (sol.top[0][1] if sol.top else [])
                variant.extend(pick)
            options_entries.append(
                {"indices": variant, "source": f"alternative in component {comp_index}"}
            )
            if len(options_entries) >= cfg.max_solution_options:
                break
        if len(options_entries) >= cfg.max_solution_options:
            break

    options: list[SolveSolution] = []
    for i, entry in enumerate(options_entries[: cfg.max_solution_options]):
        picked = [ordered[j] for j in entry["indices"]]
        obs_ids = sorted(o for c in picked for o in c.observation_ids)
        reliable = sum(weight_by_observation.get(o, 0) for o in obs_ids)
        residual = sum(c.residual_sq_us2 for c in picked)
        options.append(
            SolveSolution(
                option_id=f"OPT-{i + 1:02d}",
                candidate_ids=sorted(c.candidate_id for c in picked),
                explained_observations=len(obs_ids),
                reliable_observations=reliable,
                total_residual_sq_us2=residual,
                event_count=len(picked),
                observation_ids=obs_ids,
            )
        )

    if tie:
        notes.append(
            f"ambiguous: {len(ambiguous_components)} component(s) have disjoint "
            "decompositions that tie exactly on all objectives; a human must choose, "
            "the service will not pick one"
        )

    if not options or options[0].event_count == 0:
        return SolveResult(
            best=None,
            options=[],
            tied=False,
            proven_optimal=all_proven,
            nodes_explored=total_nodes,
            notes=notes + ["no non-empty packing found"],
            components=len(comps),
        )
    return SolveResult(
        best=options[0],
        options=options,
        tied=tie,
        proven_optimal=all_proven,
        nodes_explored=total_nodes,
        notes=notes,
        components=len(comps),
    )
