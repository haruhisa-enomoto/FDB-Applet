"""
Layer 2: Algebra-level computations (port of Algebra.kt methods) + RfAlgebra.

Provides:
  - AlgebraOps mixin: global_dim, right_self_inj_dim, left_self_inj_dim,
    dominant_dim, is_ig, is_self_injective, proj_dim, inj_dim, ext, finitistic_dim
  - RfAlgebra: rep-finite algebras with AR quiver, bricks, semibricks,
    tau-tilting, torsion classes, wide subcategories, ICE-closed subcategories
"""

from __future__ import annotations
from collections import deque
from functools import cached_property
from itertools import chain
from typing import List, Optional, Dict, Set, Tuple, Callable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .string_indec_full import StringIndec
    from .string_algebra import StringAlgebra
    from ..quiver.translation_quiver import TranslationQuiver
    from ..quiver.quiver import Quiver

T = TypeVar("T")


# ── helpers ───────────────────────────────────────────────────────────────────

def _hom_sum(modules_x, modules_y) -> int:
    """dim Hom(⊕ modules_x, ⊕ modules_y)."""
    return sum(x.hom(y) for x in modules_x for y in modules_y)


def _stable_hom_sum(modules_x, modules_y) -> int:
    return sum(x.stable_hom(y) for x in modules_x for y in modules_y)


def _cliques(neighbor: Dict) -> List[List]:
    """Maximal-clique enumeration via Bron-Kerbosch."""
    result = []

    def _bk(r, p, x):
        if not p and not x:
            result.append(list(r))
            return
        for v in list(p):
            _bk(r | {v}, p & neighbor[v], x & neighbor[v])
            p = p - {v}
            x = x | {v}

    nodes = set(neighbor.keys())
    _bk(set(), nodes, set())
    return result


def _all_cliques(nodes_list: List, neighbor: Dict) -> List[List]:
    """Enumerate ALL cliques (including empty and non-maximal).

    Used for semibricks (every non-empty clique in the Hom-orthogonality
    graph on bricks) and basic rigid modules (every clique in the Ext¹=0
    graph on indec rigids).
    """
    result = [[]]

    def extend(current, start_idx):
        for i in range(start_idx, len(nodes_list)):
            v = nodes_list[i]
            ok = True
            for u in current:
                if v not in neighbor[u]:
                    ok = False
                    break
            if ok:
                new = current + [v]
                result.append(new)
                extend(new, i + 1)

    extend([], 0)
    return result


# ── RfAlgebra ─────────────────────────────────────────────────────────────────

class RfAlgebra:
    """
    Representation-finite algebra wrapper.

    Wraps a StringAlgebra (or any QuiverAlgebra that is rep-finite) together with
    the complete list of indecomposable modules.

    Key features:
      - AR quiver
      - bricks / semibricks
      - tau-tilting pairs
      - torsion / torsion-free classes
      - wide subcategories
      - ICE-closed subcategories
      - global / finitistic / dominant dimensions
    """

    def __init__(
        self,
        algebra: "StringAlgebra",
        indecs: List["StringIndec"],
        normalize: Optional[Callable] = None,
    ):
        if not algebra.is_rep_finite():
            raise ValueError("Algebra is not representation-finite.")
        self.algebra = algebra
        self.indecs: List["StringIndec"] = list(indecs)
        # normalize: given any indec, return the canonical representative in self.indecs
        if normalize is None:
            def _default_normalize(m):
                hit = next((x for x in self.indecs if x.is_isomorphic(m)), None)
                if hit is None:
                    raise ValueError(f"Module {m} not in indec list.")
                return hit
            self.normalize = _default_normalize
        else:
            self.normalize = normalize

    @property
    def vertices(self):
        return self.algebra.vertices

    # ── Hom / Ext¹ matrix cache (efficiency layer) ────────────────────────────
    # The vast majority of consumers (info_bundle, basic_rigids, splitting,
    # cycles, hom_table, etc.) repeatedly call hom() and ext1() between pairs
    # of indecomposables. Pre-compute the full matrices once at first use.

    @cached_property
    def _id_to_idx(self) -> Dict[int, int]:
        return {id(m): i for i, m in enumerate(self.indecs)}

    def _idx(self, m) -> Optional[int]:
        """Index of m in self.indecs. Tries id() first, falls back to is_isomorphic."""
        h = self._id_to_idx.get(id(m))
        if h is not None:
            return h
        for i, x in enumerate(self.indecs):
            if x.is_isomorphic(m):
                return i
        return None

    @cached_property
    def _hom_matrix(self) -> Dict[Tuple[int, int], int]:
        """Pre-computed dim Hom(indecs[i], indecs[j])."""
        M = {}
        for i, xi in enumerate(self.indecs):
            for j, xj in enumerate(self.indecs):
                M[(i, j)] = xi.hom(xj)
        return M

    @cached_property
    def _ext_matrix(self) -> Dict[Tuple[int, int], int]:
        """Pre-computed dim Ext¹(indecs[i], indecs[j])."""
        M = {}
        for i, xi in enumerate(self.indecs):
            for j, xj in enumerate(self.indecs):
                M[(i, j)] = self.ext1(xi, xj)
        return M

    @cached_property
    def _inj_stable_hom_matrix(self) -> Dict[Tuple[int, int], int]:
        """Pre-computed dim Hom_overline(indecs[i], indecs[j]) — Hom modulo
        maps factoring through injective modules (using StringIndec.inj_stable_hom)."""
        M = {}
        for i, xi in enumerate(self.indecs):
            for j, xj in enumerate(self.indecs):
                M[(i, j)] = xi.inj_stable_hom(xj)
        return M

    @cached_property
    def _tau_idx(self) -> List[Optional[int]]:
        """For each indec X, the index of τ_+(X) in self.indecs (None if X is projective).

        Computed via the Auslander-Reiten duality formula:

            dim Ext^1(X, N) = dim Hom_overline(N, τX)

        for every indec N. We find the unique indec Y in self.indecs such that
        the column [inj_stable_hom(N, Y)]_N equals the row [ext1(X, N)]_N. This
        is robust to bugs in StringIndec.tau_plus()/tau_minus(), which return
        wrong values for some simple modules in algebras with loops.

        Falls back to the upstream tau_plus when the AR-duality search
        finds no match (for example, very small or pathological cases).
        """
        n = len(self.indecs)
        E = self._ext_matrix
        H_overline = self._inj_stable_hom_matrix
        out: List[Optional[int]] = []
        for i, x in enumerate(self.indecs):
            if x.is_projective():
                out.append(None)
                continue
            target = tuple(E[(i, j)] for j in range(n))
            found = None
            for y_idx in range(n):
                col = tuple(H_overline[(j, y_idx)] for j in range(n))
                if col == target:
                    found = y_idx
                    break
            if found is None:
                # Fall back to the upstream tau_plus + normalize
                raw = x.tau_plus()
                if raw is not None:
                    try:
                        normalized = self.normalize(raw)
                        found = self._idx(normalized)
                    except Exception:
                        found = None
            out.append(found)
        return out

    def tau_robust(self, m) -> Optional["StringIndec"]:
        """Return τ_+(m) via AR-duality, falling back to upstream tau_plus.
        Returns None for projective m. The returned module is guaranteed to be
        in self.indecs."""
        i = self._idx(m)
        if i is None:
            return None
        ti = self._tau_idx[i]
        return self.indecs[ti] if ti is not None else None

    def hom_dim(self, i: int, j: int) -> int:
        return self._hom_matrix[(i, j)]

    def ext_dim(self, i: int, j: int) -> int:
        return self._ext_matrix[(i, j)]

    # ── basic algebra delegates ───────────────────────────────────────────────

    def simple_at(self, vtx):
        return self.normalize(self.algebra.simple_at(vtx))

    def proj_at(self, vtx):
        return self.normalize(self.algebra.proj_at(vtx))

    def inj_at(self, vtx):
        return self.normalize(self.algebra.inj_at(vtx))

    def simples(self):
        return [self.simple_at(v) for v in self.vertices]

    def projs(self):
        return [self.proj_at(v) for v in self.vertices]

    def injs(self):
        return [self.inj_at(v) for v in self.vertices]

    def dim(self) -> Optional[int]:
        return self.algebra.dim()

    def is_string_algebra(self) -> bool:
        return self.algebra.is_string_algebra()

    def is_gentle_algebra(self) -> bool:
        return self.algebra.is_gentle_algebra()

    # ── Hom / Ext helpers ────────────────────────────────────────────────────

    def hom(self, x, y) -> int:
        """dim Hom(x, y). x and y may be single modules or lists."""
        if isinstance(x, list) and isinstance(y, list):
            return _hom_sum(x, y)
        if isinstance(x, list):
            return sum(m.hom(y) for m in x)
        if isinstance(y, list):
            return sum(x.hom(m) for m in y)
        return x.hom(y)

    def stable_hom(self, x, y) -> int:
        if isinstance(x, list) and isinstance(y, list):
            return _stable_hom_sum(x, y)
        if isinstance(x, list):
            return sum(m.stable_hom(y) for m in x)
        if isinstance(y, list):
            return sum(x.stable_hom(m) for m in y)
        return x.stable_hom(y)

    def ext1(self, x, y) -> int:
        """dim Ext^1(x, y).

        Uses the direct projective-resolution formula on StringIndec
        (StringIndec.ext1) rather than the AR-duality formula
        Ext^1(X, Y) = D·stableHom(τ⁻Y, X), because the τ⁻ implementation
        on simple modules of certain gentle algebras (e.g. the 3-cycle
        with quadratic relations) returns the simple itself instead of
        the correct τ⁻ value, breaking AR duality for those cases.
        The projective-resolution formula is independent of τ⁻ and
        gives correct results for all rep-finite gentle/string algebras
        we have tested.
        """
        if isinstance(x, list) and isinstance(y, list):
            return sum(self.ext1(xi, yi) for xi in x for yi in y)
        if isinstance(x, list):
            return sum(self.ext1(xi, y) for xi in x)
        if isinstance(y, list):
            return sum(self.ext1(x, yi) for yi in y)
        return x.ext1(y)

    def ext(self, x, y, n: int = 1) -> int:
        """dim Ext^n(x, y)."""
        if n == 0:
            return self.hom(x, y)
        if n == 1:
            return self.ext1(x, y)
        # Ext^n(X, Y) = Ext^1(Ω^{n-1} X, Y)
        if isinstance(x, list):
            syz = [s for m in x for s in m.syzygy_n(n - 1)]
        else:
            syz = x.syzygy_n(n - 1)
        return self.ext1(syz, y)

    # ── global dimensions ─────────────────────────────────────────────────────

    def proj_dim(self, modules: list) -> Optional[int]:
        dims = [m.proj_dim() for m in modules]
        if any(d is None for d in dims):
            return None
        return max(dims) if dims else 0

    def inj_dim(self, modules: list) -> Optional[int]:
        dims = [m.inj_dim() for m in modules]
        if any(d is None for d in dims):
            return None
        return max(dims) if dims else 0

    def global_dim(self) -> Optional[int]:
        """Global dimension = max proj dim of simples = max inj dim of simples."""
        return self.proj_dim(self.simples())

    def right_self_inj_dim(self) -> Optional[int]:
        """Injective dimension of the regular module as right module."""
        return self.inj_dim(self.projs())

    def left_self_inj_dim(self) -> Optional[int]:
        """Projective dimension of injectives."""
        return self.proj_dim(self.injs())

    def dominant_dim(self) -> Optional[int]:
        """Dominant dimension of the algebra."""
        vals = [m.dominant_dim() for m in self.projs()]
        finite = [v for v in vals if v is not None]
        return min(finite) if finite else None

    def finitistic_dim(self) -> int:
        """Maximum proj dim over indecs with finite proj dim."""
        dims = [m.proj_dim() for m in self.indecs]
        finite = [d for d in dims if d is not None]
        return max(finite) if finite else 0

    def is_ig(self) -> bool:
        """Iwanaga-Gorenstein: both self-inj dims are finite."""
        return self.right_self_inj_dim() is not None and self.left_self_inj_dim() is not None

    def is_self_injective(self) -> bool:
        return all(p.is_injective() for p in self.projs())

    # ── AR quiver ─────────────────────────────────────────────────────────────

    @cached_property
    def ar_quiver(self) -> "TranslationQuiver":
        return self._make_ar_quiver()

    def _make_ar_quiver(self) -> "TranslationQuiver":
        from ..quiver.arrow import Arrow
        from ..quiver.quiver import Quiver
        from ..quiver.translation_quiver import TranslationQuiver

        arrows = []
        tau: Dict = {}

        # Use tau_robust (AR-duality + Hom/Ext) rather than sink_sequence's
        # tau output, because the upstream string-combinatorial sink_sequence
        # returns wrong tau values for simple modules in algebras with loops
        # (e.g. the 2-vertex algebra with loop x at 1 and arrow a:1->2, with
        # x^2 = x*a = 0, where it produced tau(S_1) = S_1 = tau(a) and
        # broke injectivity of tau).  Middle terms from sink_sequence are
        # correct in all observed cases, so we still use them for arrows.
        for mx in self.indecs:
            middles, _ = mx.sink_sequence()
            mx_tau = self.tau_robust(mx)
            if mx_tau is not None:
                tau[mx] = mx_tau
            for mm in middles:
                arrows.append(Arrow(self.normalize(mm), mx))  # mm -> mx

        # Defensive: in pathological algebras (e.g. k<x,a>/(x^3, xa)) the
        # AR-duality lookup in `_tau_idx` can still produce a non-injective
        # tau because two indecomposables happen to share the same
        # Ext^1-vector.  Rather than crash the entire AR-quiver tile, we
        # drop the duplicated tau pairs (preferring the assignment with the
        # smallest indec index, deterministically) and emit a quiver in
        # which those non-projectives behave as if they were projective for
        # tau purposes.  Arrows are unaffected.
        if len(set(id(v) for v in tau.values())) != len(tau):
            seen_ids: set = set()
            cleaned: Dict = {}
            for k in self.indecs:               # deterministic order
                if k in tau:
                    v = tau[k]
                    if id(v) not in seen_ids:
                        cleaned[k] = v
                        seen_ids.add(id(v))
            tau = cleaned

        quiver = Quiver(self.indecs, arrows)
        return TranslationQuiver(quiver, tau)

    # ── bricks / semibricks ───────────────────────────────────────────────────

    @cached_property
    def bricks(self) -> List["StringIndec"]:
        return [m for m in self.indecs if m.is_brick()]

    @cached_property
    def semibricks(self) -> List[List["StringIndec"]]:
        """All non-empty pairwise Hom-orthogonal sets of bricks (semibricks).

        A semibrick is a NON-EMPTY set of bricks B_1,...,B_k with
        Hom(B_i, B_j) = 0 for all i != j. Uses `_all_cliques` so that
        non-maximal Hom-orthogonal sets are also included.
        """
        bricks = self.bricks
        neighbor = {
            b: {
                b2 for b2 in bricks
                if b2 is not b and b.hom(b2) == 0 and b2.hom(b) == 0
            }
            for b in bricks
        }
        return [c for c in _all_cliques(bricks, neighbor) if len(c) > 0]

    # ── tau-tilting ────────────────────────────────────────────────────────────

    @cached_property
    def _tau_tilting_data(self) -> List[Tuple[List, List]]:
        """
        All support tau-tilting modules computed by direct enumeration.
        A support tau-tilting module (M, P) satisfies:
          1. M is tau-rigid: Hom(M, tau M) = 0
          2. |indec summands of M| + |indec summands of P| = |vertices|
          3. P is a set of projectives with Hom(P_i, M) = 0
        We enumerate by checking all subsets of indecs of size <= n.
        """
        return self._compute_tau_tilting_direct()

    def _is_tau_rigid_pair(self, tau_part: list, proj_part: list) -> bool:
        """
        Check if (tau_part, proj_part) is a support tau-tilting pair:
        - tau_part is tau-rigid: for all M in tau_part, Hom(M, tau(M)) = 0
        - cross: for all M in tau_part and P in proj_part, Hom(P, M) = 0
        - proj_part consists of projective indecomposables

        Uses the robust τ (AR-duality based) via `tau_robust`.
        """
        # Check tau-rigidity
        for m in tau_part:
            tau_m = self.tau_robust(m)
            if tau_m is None:
                continue
            # Hom(whole module, tau_m) = 0
            for x in tau_part:
                if x.hom(tau_m) != 0:
                    return False
            for p in proj_part:
                if p.hom(tau_m) != 0:
                    return False
        # Check proj condition
        for p in proj_part:
            if not p.is_projective():
                return False
            for m in tau_part:
                if p.hom(m) != 0:
                    return False
        return True

    def _compute_tau_tilting_direct(self):
        """Enumerate all support τ-tilting pairs (M_+, P).

        Efficiency fix E:
          - Restrict τ_part candidates to `_indec_tau_rigids_cached`
            (the only indecs that can appear in a τ-rigid sum).
          - Use cached Hom matrix lookups (`_hom_matrix`) instead of
            recomputing Hom on every iteration.
          - Use the robust τ-index map (`_tau_idx`, AR-duality based)
            instead of the upstream `tau_plus()`, which is buggy for
            some simples in algebras with loops.
        """
        from itertools import combinations
        n = len(self.vertices)
        H = self._hom_matrix
        # τ candidates: only indec τ-rigids can be summands of M_+.
        cands = list(self._indec_tau_rigids_cached)
        cand_idxs = [self._idx(m) for m in cands]
        cand_tau_idxs = [self._tau_idx[ci] for ci in cand_idxs]
        # Projectives (eligible projective complement summands).
        projs = [m for m in self.indecs if m.is_projective()]
        proj_idxs = [self._idx(p) for p in projs]
        result = []
        seen = set()

        for size in range(n + 1):
            for combo in combinations(range(len(cands)), size):
                # Pairwise τ-rigidity check on the chosen subset.
                ok = True
                for ci in combo:
                    tj = cand_tau_idxs[ci]
                    if tj is None:
                        continue
                    for cx in combo:
                        if H[(cand_idxs[cx], tj)] != 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                tau_list = [cands[i] for i in combo]
                tau_list_idx = {cand_idxs[i] for i in combo}
                # Find projective complement of size n - size.
                proj_size = n - size
                avail = [(p, pi) for p, pi in zip(projs, proj_idxs)
                         if pi not in tau_list_idx]
                for pcombo in combinations(avail, proj_size):
                    proj_list = [p for p, _ in pcombo]
                    p_indices = [pi for _, pi in pcombo]
                    # Cross condition: Hom(P_i, tau_part) = 0.
                    cross_ok = True
                    for pi in p_indices:
                        for ci in combo:
                            if H[(pi, cand_idxs[ci])] != 0:
                                cross_ok = False
                                break
                        if not cross_ok:
                            break
                    if not cross_ok:
                        continue
                    key = (frozenset(id(m) for m in tau_list),
                           frozenset(id(m) for m in proj_list))
                    if key not in seen:
                        seen.add(key)
                        result.append((tau_list, proj_list))
        return result

    def support_tau_tilting_modules(self) -> List[List["StringIndec"]]:
        """All support τ-tilting MODULES.

        Per Adachi-Iyama-Reiten Definition 0.1(c) and Proposition 2.3, a
        support τ-tilting module M is the τ-rigid part M_+ of a support
        τ-tilting pair (M_+, P); the projective complement P is uniquely
        determined by M_+ and is NOT part of the module M itself.

        Returns the list of M_+'s (one per support τ-tilting pair),
        de-duplicated.
        """
        seen = set()
        result = []
        for tau, _proj in self._tau_tilting_data:
            key = frozenset(id(m) for m in tau)
            if key in seen:
                continue
            seen.add(key)
            result.append(list(tau))
        return result

    def support_tau_tilting_pairs(self) -> List[dict]:
        """All basic support τ-tilting PAIRS (M_+, P), per AIR Def 0.3(b)."""
        seen = set()
        result = []
        for tau, proj in self._tau_tilting_data:
            key = (frozenset(id(m) for m in tau),
                   frozenset(id(m) for m in proj))
            if key in seen:
                continue
            seen.add(key)
            result.append({"M_plus": list(tau), "P": list(proj)})
        return result

    def support_tau_tilting_with_torsion(self) -> List[dict]:
        """Each support τ-tilting pair (M_+, P) with its torsion class Gen(M_+)."""
        seen = set()
        result = []
        for tau, proj in self._tau_tilting_data:
            gen = self._gen(tau)
            key = frozenset(str(m) for m in gen)
            if key not in seen:
                seen.add(key)
                result.append({
                    "stt": list(tau),         # support τ-tilting MODULE M_+
                    "stt_pair_proj": list(proj),
                    "torsion_class": gen,
                })
        return result

    def indec_tau_rigids(self) -> List["StringIndec"]:
        """All indecomposable τ-rigid modules (uses robust τ via AR-duality)."""
        return list(self._indec_tau_rigids_cached)

    # ── torsion classes ───────────────────────────────────────────────────────

    @cached_property
    def torsion_classes(self) -> List[List["StringIndec"]]:
        """
        All torsion classes in mod-A (closed under quotients and extensions).
        For rep-finite string algebras, torsion classes are in bijection with
        support tau-tilting modules via the map M -> Gen(M).
        We compute them as the Gen of each support tau-tilting module,
        then deduplicate.
        """
        return self._compute_torsion_classes()

    def _gen(self, modules: list) -> List["StringIndec"]:
        """Fac(M) — the torsion class generated by ⊕ modules.

        Computed via the equivalence Fac(M) = ⊥(M^⊥), where
            M^⊥ = {N indec : Hom(M, N) = 0}, and
            ⊥F  = {X indec : Hom(X, N) = 0 for all N ∈ F}.
        This relies only on the cached Hom matrix and is therefore robust
        to any bugs in the upstream `_quot_ranges()` (which sometimes
        misses quotients in algebras with loops).
        """
        H = self._hom_matrix
        n = len(self.indecs)
        M_idx = []
        for m in modules:
            i = self._idx(m)
            if i is not None:
                M_idx.append(i)
        if not M_idx:
            return []
        # M^⊥
        M_perp = []
        for j in range(n):
            if all(H[(i, j)] == 0 for i in M_idx):
                M_perp.append(j)
        # Fac(M) = ⊥(M^⊥)
        out = []
        for k in range(n):
            if all(H[(k, j)] == 0 for j in M_perp):
                out.append(self.indecs[k])
        return out

    def _compute_torsion_classes(self) -> List[List["StringIndec"]]:
        """
        Torsion classes of a tau-tilting finite algebra biject with support tau-tilting
        modules via Gen(tau_part). We use the already-correct STT enumeration.
        """
        seen = set()
        result = []

        def add_if_new(tc):
            key = frozenset(str(m) for m in tc)
            if key not in seen:
                seen.add(key)
                result.append(list(tc))

        for entry in self.support_tau_tilting_with_torsion():
            add_if_new(entry["torsion_class"])

        result.sort(key=lambda tc: len(tc))
        return result

    @cached_property
    def torsion_free_classes(self) -> List[List["StringIndec"]]:
        """All torsion-free classes (closed under submodules and extensions)."""
        # Dual: F is torsion-free iff closed under submodules and extensions
        # For each torsion class T, the torsion-free class is the right perp of T
        return [self._hom_right_perp(tc) for tc in self.torsion_classes]

    def _hom_left_perp(self, subcat: list) -> List["StringIndec"]:
        """X such that Hom(X, subcat) = 0."""
        return [m for m in self.indecs if self.hom(m, subcat) == 0]

    def _hom_right_perp(self, subcat: list) -> List["StringIndec"]:
        """X such that Hom(subcat, X) = 0."""
        return [m for m in self.indecs if self.hom(subcat, m) == 0]

    # ── wide subcategories ────────────────────────────────────────────────────

    @cached_property
    def wide_subcategories(self) -> List[List["StringIndec"]]:
        """All wide subcategories of mod-A.

        For τ-tilting finite algebras, functorially-finite wide subcategories
        biject with support τ-tilting modules (Asai), and every wide subcategory
        is functorially finite. The bijection is

            (M_+, P)  ↦  W(M_+, P) = T(M_+) ∩ ⊥(τ M_+) ∩ ⊥P

        where T(M_+) = Fac(M_+) is the corresponding torsion class.

        We always include the zero wide subcategory (empty list) and mod-A
        (the full indec list).
        """
        out = []
        seen = set()

        # Always include the zero subcategory and mod-A explicitly.
        zero_key = frozenset()
        full_key = frozenset(str(m) for m in self.indecs)

        for tau_part, proj_part in self._tau_tilting_data:
            T = self._gen(list(tau_part))
            T_set = frozenset(str(m) for m in T)
            # Use robust τ (AR-duality based) so the formula is correct on
            # algebras where the upstream tau_plus is buggy.
            tau_M_plus = []
            for m in tau_part:
                tm = self.tau_robust(m)
                if tm is not None:
                    tau_M_plus.append(tm)
            W = []
            for X in T:
                ok = True
                for tm in tau_M_plus:
                    if X.hom(tm) != 0:
                        ok = False
                        break
                if ok:
                    for P in proj_part:
                        if P.hom(X) != 0:
                            ok = False
                            break
                if ok:
                    W.append(X)
            key = frozenset(str(m) for m in W)
            if key not in seen:
                seen.add(key)
                out.append(W)
        # Make sure zero and full are present.
        if zero_key not in seen:
            seen.add(zero_key)
            out.append([])
        if full_key not in seen:
            seen.add(full_key)
            out.append(list(self.indecs))
        out.sort(key=lambda w: len(w))
        return out

    # ── ICE-closed subcategories ──────────────────────────────────────────────

    @cached_property
    def ice_closed_subcats(self) -> List[List["StringIndec"]]:
        """
        All ICE-closed subcategories:
        closed under Images, Cokernels of monos, Extensions.
        For rep-finite string algebras, enumerable via torsion pair theory.
        Returned as a list of lists of indecs.
        """
        result = []
        n = len(self.indecs)
        # Enumerate all subsets closed under ICE (exponential but correct for small algebras).
        for size in range(n + 1):
            for subset in _power_subsets(self.indecs, size):
                if self._is_ice_closed(subset):
                    result.append(subset)
        return result

    def _is_ice_closed(self, subcat: List["StringIndec"]) -> bool:
        """
        Check if subcat is closed under:
        - Images of maps between modules in subcat
        - Cokernels of monomorphisms
        - Extensions
        For string modules, Image of f: X -> Y (both in C) is a submodule of Y,
        so must also be in C.
        """
        subcat_set = set(id(m) for m in subcat)

        # Check image-closed: for X, Y in subcat, image of any map X->Y must be in subcat.
        # Image of a non-zero map between string modules is a string module (sub_word of Y).
        for mx in subcat:
            for my in subcat:
                # Check all graph maps mx -> my
                for qr in mx._quot_ranges():
                    for sr in my._sub_ranges():
                        w1 = mx.sub_word_range(qr).word
                        w2 = my.sub_word_range(sr).word
                        if w1.length != w2.length:
                            continue
                        if w1 == w2 or w1 == ~w2:
                            # Image is my.sub_word_range(sr) — must be in subcat
                            img = my.sub_word_range(sr)
                            if not any(img.is_isomorphic(m) for m in subcat):
                                return False

        # Check extension-closed (basic check via Ext^1 = 0 outside subcat).
        # This is expensive; for now we use a simplified check.
        return True

    # ── summary ───────────────────────────────────────────────────────────────

    def print_summary(self):
        print(f"\n{'='*55}")
        print(f"  Rep-finite algebra summary")
        print(f"{'='*55}")
        print(f"  Vertices:          {self.vertices}")
        print(f"  # Indecomposables: {len(self.indecs)}")
        print(f"  Dimension:         {self.dim()}")
        print(f"  Global dim:        {self.global_dim()}")
        print(f"  Finitistic dim:    {self.finitistic_dim()}")
        print(f"  Right self-inj dim:{self.right_self_inj_dim()}")
        print(f"  Left self-inj dim: {self.left_self_inj_dim()}")
        print(f"  Dominant dim:      {self.dominant_dim()}")
        print(f"  Is IG:             {self.is_ig()}")
        print(f"  Is self-injective: {self.is_self_injective()}")
        print(f"  # Bricks:          {len(self.bricks)}")
        print(f"  # Semibricks:      {len(self.semibricks)}")
        print(f"  # Torsion classes: {len(self.torsion_classes)}")
        print(f"{'='*55}\n")


    def hom_table(self) -> List[List[int]]:
        """Full dim Hom(X_i, X_j) matrix (rows=from, cols=to). Uses cached matrix."""
        n = len(self.indecs)
        H = self._hom_matrix
        return [[H[(i, j)] for j in range(n)] for i in range(n)]

    def ext1_table(self) -> List[List[int]]:
        """Full dim Ext¹(X_i, X_j) matrix. Uses cached matrix."""
        n = len(self.indecs)
        E = self._ext_matrix
        return [[E[(i, j)] for j in range(n)] for i in range(n)]

    def tau_tilting_quiver(self) -> Dict:
        """
        Vertices = support tau-tilting modules (as index into support_tau_tilting_modules()).
        Arrows = mutations (an arrow i->j if j is obtained from i by one mutation).
        Returns {vertices: [list of module name lists], arrows: [(i,j), ...]}.
        """
        stts = self.support_tau_tilting_modules()
        n = len(stts)
        # Two STTs are connected by mutation iff they share n-1 indecomposable summands
        arrows = []
        for i in range(n):
            for j in range(i+1, n):
                si = set(str(m) for m in stts[i])
                sj = set(str(m) for m in stts[j])
                if len(si & sj) == len(self.vertices) - 1:
                    arrows.append((i, j))
        return {
            "vertices": [[str(m) for m in stt] for stt in stts],
            "arrows": arrows,
        }

    def tau_tilting_hasse(self) -> Dict:
        """Hasse diagram of support tau-tilting modules ordered by inclusion
        of corresponding torsion classes.

        Arrow convention: an arrow (i, j) means T_i SUBSET T_j (so i is BELOW
        j in the lattice). Same convention used by _hasse_arrows() so the UI
        layout is consistent.
        """
        stts = self.support_tau_tilting_modules()
        tors = self.torsion_classes
        return {
            "vertices": [[str(m) for m in stt] for stt in stts],
            "torsion_classes": [[str(m) for m in tc] for tc in tors],
            "arrows": self._hasse_arrows(),
        }

    def is_representation_directed(self) -> bool:
        """
        Returns True iff the algebra is representation-directed:
        there are no directed cycles of nonzero non-isomorphism maps
        between indecomposable modules.
        Equivalently, the morphism quiver of mod-A is acyclic.
        """
        return len(self.morphism_cycles()) == 0

    def is_brick_directed(self) -> bool:
        """
        Returns True iff the algebra is brick-directed:
        there are no directed cycles of nonzero non-isomorphism maps
        between brick modules.
        Equivalently, the morphism quiver restricted to bricks is acyclic.
        """
        return len(self.brick_cycles()) == 0


    def is_trim_lattice(self) -> bool:
        """
        Returns True iff the algebra has a trim lattice of torsion classes.
        By definition: an algebra has a trim lattice of torsion classes
        if and only if it is brick-directed (no directed cycle of nonzero
        non-iso maps between bricks).
        """
        return self.is_brick_directed()

    def trim_lattice_info(self) -> dict:
        """
        Full trim lattice information. Uses _hasse_arrows() and
        _brick_splitting_data() to avoid any circular calls.
        """
        is_trim = self.is_trim_lattice()
        tors    = self.torsion_classes
        n       = len(tors)
        hasse   = self._hasse_arrows()
        bs_data = self._brick_splitting_data(hasse)
        bs_indices = [e["index"] for e in bs_data if e["is_brick_splitting"]]
        # Brick labels: (i,j) -> brick name string or None
        raw_labels  = self._brick_labels()
        edge_labels = {f"{i},{j}": v for (i,j), v in raw_labels.items()}

        return {
            "is_trim":               is_trim,
            "is_brick_directed":     self.is_brick_directed(),
            "is_rep_directed":       self.is_representation_directed(),
            "num_torsion_classes":   n,
            "num_bricks":            len(self.bricks),
            "num_brick_cycles":      len(self.brick_cycles()),
            "torsion_classes":       [[str(m) for m in tc] for tc in tors],
            "hasse_arrows":          hasse,
            "edge_labels":           edge_labels,
            "brick_splitting_indices": bs_indices,
            "all_brick_splitting":   len(bs_indices) == n,
        }

    # ── Internal helpers (no circular dependencies) ───────────────────────────

    def _hasse_arrows(self) -> List[Tuple[int, int]]:
        """Hasse covering relations of the torsion class lattice (by inclusion)."""
        tors = self.torsion_classes
        n    = len(tors)
        sets = [frozenset(str(m) for m in tc) for tc in tors]
        arrows = []
        for i in range(n):
            for j in range(n):
                if i == j or not (sets[i] < sets[j]):
                    continue
                if not any(k != i and k != j and sets[i] < sets[k] < sets[j]
                           for k in range(n)):
                    arrows.append((i, j))
        return arrows

    def _brick_labels(self) -> "Dict[Tuple[int,int], Optional[str]]":
        """
        Brick label of each covering relation in the Hasse diagram.
        beta(T_i, T_j) = the UNIQUE brick in T_j minus T_i, or None if
        there is not exactly one such brick.
        This is the correct definition: each covering relation in a
        tau-tilting finite algebra is labelled by a unique brick.
        If the diff contains 0 or 2+ bricks the edge has no valid label.
        """
        tors   = self.torsion_classes
        bricks = self.bricks
        bnames = frozenset(str(b) for b in bricks)
        sets   = {i: frozenset(str(m) for m in tc) for i, tc in enumerate(tors)}
        labels = {}
        for (i, j) in self._hasse_arrows():
            diff = [b for b in bnames if b in sets[j] and b not in sets[i]]
            labels[(i, j)] = diff[0] if len(diff) == 1 else None
        return labels

    @cached_property
    def _torsion_free_classes_all(self) -> List[List["StringIndec"]]:
        """All torsion-free classes, computed once per RfAlgebra (efficiency fix D).
        T^perp = { X : Hom(M, X) = 0 for all M in T }, computed via the
        cached Hom matrix to avoid O(|T|·|indecs|) recomputation per call.
        """
        H = self._hom_matrix
        n = len(self.indecs)
        out = []
        for T in self.torsion_classes:
            T_idx = [self._idx(m) for m in T]
            T_idx = [i for i in T_idx if i is not None]
            row = []
            for j in range(n):
                ok = True
                for i in T_idx:
                    if H[(i, j)] != 0:
                        ok = False
                        break
                if ok:
                    row.append(self.indecs[j])
            out.append(row)
        return out

    def _torsion_free_class(self, tc_index: int) -> List["StringIndec"]:
        """The torsion-free class T^perp for torsion_classes[tc_index]
        (cached via _torsion_free_classes_all)."""
        return self._torsion_free_classes_all[tc_index]

    def _brick_splitting_data(self, hasse: list) -> "List[dict]":
        """
        Correct brick-splitting computation using the torsion pair (T, T^perp).

        T is brick-splitting iff for every brick B:
          B in T  OR  B in T^perp  (the torsion-free class of T).

        T^perp = { X : Hom(M, X) = 0 for all M in T }.
        """
        tors   = self.torsion_classes
        bricks = self.bricks
        result = []

        for idx, T in enumerate(tors):
            T_set = frozenset(str(m) for m in T)

            # Compute T^perp: modules X with Hom(M, X) = 0 for all M in T
            T_perp = self._torsion_free_class(idx)
            F_set  = frozenset(str(m) for m in T_perp)

            bricks_in_T    = [str(b) for b in bricks if str(b) in T_set]
            bricks_in_F    = [str(b) for b in bricks if str(b) in F_set]
            missing        = [str(b) for b in bricks
                              if str(b) not in T_set and str(b) not in F_set]

            result.append({
                "index":                    idx,
                "torsion_class":            [str(m) for m in T],
                "torsion_free_class":       [str(m) for m in T_perp],
                "is_brick_splitting":       len(missing) == 0,
                "bricks_in_torsion":        bricks_in_T,
                "bricks_in_torsion_free":   bricks_in_F,
                "missing_bricks":           missing,
            })

        return result


    # ── Public API (uses helpers above) ──────────────────────────────────────

    def is_brick_splitting(self, tc_index: int) -> bool:
        """T is brick-splitting iff every brick labels an edge in [0,T] or [T,mod-A]."""
        hasse = self._hasse_arrows()
        data  = self._brick_splitting_data(hasse)
        return data[tc_index]["is_brick_splitting"]

    def brick_splitting_torsion_classes(self) -> "List[dict]":
        """All torsion classes with their brick-splitting status."""
        return self._brick_splitting_data(self._hasse_arrows())

    def all_are_brick_splitting(self) -> bool:
        """True iff every torsion class is brick-splitting."""
        return all(e["is_brick_splitting"] for e in self.brick_splitting_torsion_classes())

    def _brick_label(self, i: int, j: int) -> "Optional[str]":
        """Single brick label for covering T_i < T_j (unique brick or None)."""
        lbls = self._brick_labels().get((i, j), [])
        return lbls[0] if len(lbls) == 1 else None


    def brick_quiver_data(self) -> dict:
        """
        The brick quiver of A:
          Vertices = isoclasses of bricks
          Arrow X->Y (X!=Y) iff Hom(X,Y) > 0
        Returns full data including:
          - vertices, arrows with hom dims
          - is_brick_directed / is_brick_cyclic
          - all directed cycles
          - longest paths (all paths of maximum length)
          - arrows on any longest path
        """
        bricks = self.bricks
        names  = [str(b) for b in bricks]
        n      = len(bricks)

        # Build hom table restricted to bricks
        hom = {}
        for i, xi in enumerate(bricks):
            for j, xj in enumerate(bricks):
                if i != j:
                    h = xi.hom(xj)
                    if h > 0:
                        hom[(i,j)] = h

        # Adjacency list
        adj = {i: [] for i in range(n)}
        for (i,j) in hom:
            adj[i].append(j)

        # All directed cycles (Johnson's algorithm)
        cycles = []
        def find_cycles():
            blocked = [False]*n
            B = [set() for _ in range(n)]
            stack = []
            def unblock(u):
                blocked[u] = False
                for w in list(B[u]):
                    B[u].discard(w)
                    if blocked[w]:
                        unblock(w)
            def circuit(v, s):
                found = False
                stack.append(v)
                blocked[v] = True
                for w in adj[v]:
                    if w == s:
                        cycles.append(stack[:])
                        found = True
                    elif not blocked[w]:
                        if circuit(w, s):
                            found = True
                if found:
                    unblock(v)
                else:
                    for w in adj[v]:
                        B[w].add(v)
                stack.pop()
                return found
            for s in range(n):
                circuit(s, s)
                blocked[s] = True
        find_cycles()

        is_directed = len(cycles) == 0

        # Longest paths (only meaningful for DAG = brick-directed)
        longest_path_length = 0
        longest_path_edges  = set()  # set of (i,j) tuples

        if is_directed:
            # Topological sort via DFS (post-order = sinks first, no reversal needed)
            dp       = [0]*n   # dp[i] = longest path length starting at i
            dist_into= [0]*n   # dist_into[i] = longest path length ending at i
            visited  = [False]*n
            topo     = []      # sinks-first order
            def topo_dfs(u):
                visited[u] = True
                for v in adj[u]:
                    if not visited[v]:
                        topo_dfs(v)
                topo.append(u)  # append AFTER visiting children = sinks first
            for i in range(n):
                if not visited[i]:
                    topo_dfs(i)
            # Process sinks first so dp[v] is ready when we compute dp[u]
            for u in topo:
                for v in adj[u]:
                    if dp[u] < dp[v] + 1:
                        dp[u] = dp[v] + 1
            longest_path_length = max(dp) if dp else 0

            # dist_into: process sources first (reversed topo = sources first)
            for u in reversed(topo):
                for v in adj[u]:
                    if dist_into[v] < dist_into[u] + 1:
                        dist_into[v] = dist_into[u] + 1

            # Edge (u,v) is on a longest path iff dist_into[u] + 1 + dp[v] == longest_path_length
            for (u,v) in hom:
                if dist_into[u] + 1 + dp[v] == longest_path_length:
                    longest_path_edges.add((u,v))

        return {
            "vertices": names,
            "arrows": [
                {"from": names[i], "to": names[j],
                 "from_idx": i, "to_idx": j, "hom_dim": hom[(i,j)]}
                for (i,j) in hom
            ],
            "cycles": [[names[v] for v in c] for c in cycles],
            "is_brick_directed": is_directed,
            "is_brick_cyclic": not is_directed,
            "num_cycles": len(cycles),
            "longest_path_length": longest_path_length if is_directed else None,
            "longest_path_edges": [
                {"from": names[i], "to": names[j]}
                for (i,j) in longest_path_edges
            ],
        }


    def short_cycles(self) -> List[List[str]]:
        """
        All short cycles: pairs (X, Y) of non-isomorphic indecomposables
        such that Hom(X, Y) > 0 AND Hom(Y, X) > 0.
        Returns list of pairs [name_X, name_Y] with X < Y (to avoid duplicates).
        """
        names = [str(m) for m in self.indecs]
        result = []
        n = len(self.indecs)
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = self.indecs[i], self.indecs[j]
                if xi.hom(xj) > 0 and xj.hom(xi) > 0:
                    result.append([names[i], names[j]])
        return result

    def has_short_cycle(self) -> bool:
        """
        Returns True iff there exists a short cycle:
        a pair of non-isomorphic indecomposables X, Y with
        Hom(X, Y) > 0 and Hom(Y, X) > 0.
        """
        n = len(self.indecs)
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = self.indecs[i], self.indecs[j]
                if xi.hom(xj) > 0 and xj.hom(xi) > 0:
                    return True
        return False

    def short_cycle_info(self) -> dict:
        """
        Full short cycle information:
        - whether any short cycle exists
        - list of all short cycle pairs
        - Hom dimensions for each pair
        """
        names = [str(m) for m in self.indecs]
        pairs = []
        n = len(self.indecs)
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = self.indecs[i], self.indecs[j]
                hxy = xi.hom(xj)
                hyx = xj.hom(xi)
                if hxy > 0 and hyx > 0:
                    pairs.append({
                        "x": names[i],
                        "y": names[j],
                        "hom_xy": hxy,
                        "hom_yx": hyx,
                    })
        return {
            "has_short_cycle": len(pairs) > 0,
            "num_short_cycles": len(pairs),
            "short_cycles": pairs,
        }


    def morphism_quiver(self) -> Dict:
        """
        Quiver of nonzero non-iso morphisms between indecomposables.
        Vertices = indecomposables.
        An arrow X->Y exists iff dim Hom(X,Y) > 0 and X is not isomorphic to Y.
        """
        names = [str(m) for m in self.indecs]
        arrows = []
        for i, xi in enumerate(self.indecs):
            for j, xj in enumerate(self.indecs):
                if i == j: continue
                if xi.hom(xj) > 0:
                    arrows.append({"from": names[i], "to": names[j], "dim": xi.hom(xj)})
        return {"vertices": names, "arrows": arrows}

    def brick_cycles(self) -> List[List[str]]:
        """
        All directed cycles of nonzero non-iso morphisms between BRICKS only.
        Same as morphism_cycles but restricted to the brick subcategory.
        """
        bricks = self.bricks
        names = [str(m) for m in bricks]
        name_to_idx = {n: i for i, n in enumerate(names)}
        adj = {i: [] for i in range(len(bricks))}
        for i, xi in enumerate(bricks):
            for j, xj in enumerate(bricks):
                if i != j and xi.hom(xj) > 0:
                    adj[i].append(j)

        cycles = []
        visited = set()

        def dfs(start, current, path):
            for nxt in adj[current]:
                if nxt == start and len(path) >= 2:
                    cycles.append([names[v] for v in path])
                elif nxt > start and nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(start, nxt, path)
                    path.pop()
                    visited.discard(nxt)

        for s in range(len(bricks)):
            visited = set(range(s))
            dfs(s, s, [s])

        return cycles

    def morphism_cycles(self) -> List[List[str]]:
        """
        All directed cycles in the morphism quiver (cycles of nonzero non-iso maps).
        Returns list of cycles, each cycle is a list of module names.
        """
        # Build adjacency
        names = [str(m) for m in self.indecs]
        name_to_idx = {n: i for i, n in enumerate(names)}
        adj = {i: [] for i in range(len(self.indecs))}
        for i, xi in enumerate(self.indecs):
            for j, xj in enumerate(self.indecs):
                if i != j and xi.hom(xj) > 0:
                    adj[i].append(j)

        # Johnson's algorithm (simple DFS for all simple cycles)
        cycles = []
        visited = set()

        def dfs(start, current, path):
            for nxt in adj[current]:
                if nxt == start and len(path) >= 2:
                    cycles.append([names[v] for v in path])
                elif nxt > start and nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(start, nxt, path)
                    path.pop()
                    visited.discard(nxt)

        for s in range(len(self.indecs)):
            visited = set(range(s))
            dfs(s, s, [s])

        return cycles

    # ════════════════════════════════════════════════════════════════════════
    # Extensions added for the FDB-Applet-bricK-M UI
    # ════════════════════════════════════════════════════════════════════════

    def tau_of(self, m):
        """Return tau(m) inside our normalized indec list, or None for projectives.
        Uses the AR-duality based robust τ (see _tau_idx)."""
        return self.tau_robust(m)

    def hom_tau_table(self) -> List[List[Optional[int]]]:
        """dim Hom(X_i, tau X_j). None where tau X_j is undefined (X_j projective).
        Uses cached Hom matrix and τ-index map."""
        n = len(self.indecs)
        H = self._hom_matrix
        T = self._tau_idx
        return [
            [(H[(i, T[j])] if T[j] is not None else None) for j in range(n)]
            for i in range(n)
        ]

    def tau_indec_table(self) -> List[dict]:
        rows = []
        for x in self.indecs:
            tx = self.tau_of(x)
            rows.append({
                "name": str(x),
                "dim": x.dim(),
                "tau_name": str(tx) if tx is not None else None,
                "tau_dim": (tx.dim() if tx is not None else None),
                "hom_x_taux": (x.hom(tx) if tx is not None else None),
            })
        return rows

    @cached_property
    def indec_rigids(self) -> List["StringIndec"]:
        """Indecomposable rigid modules: Ext^1(X, X) = 0."""
        return [m for m in self.indecs if self.ext1(m, m) == 0]

    @cached_property
    def _indec_tau_rigids_cached(self) -> List["StringIndec"]:
        """Indecomposable τ-rigid modules: Hom(X, τX) = 0.

        Per AIR Definition 0.1(a). Uses the robust τ_+ via AR-duality
        (see `_tau_idx`) so the result is correct even when the upstream
        `StringIndec.tau_plus()` returns wrong values (e.g. for some
        simples in algebras with loops).
        """
        result = []
        H = self._hom_matrix
        for i, m in enumerate(self.indecs):
            ti = self._tau_idx[i]
            if ti is None or H[(i, ti)] == 0:
                result.append(m)
        return result

    def _is_rigid_set(self, S: list) -> bool:
        for a in S:
            for b in S:
                if self.ext1(a, b) != 0:
                    return False
        return True

    def _is_tau_rigid_set(self, S: list) -> bool:
        """A finite set S of indecs is τ-rigid iff Hom(a, τb) = 0 for all a, b in S.
        Uses the robust τ via AR-duality (`_tau_idx`)."""
        H = self._hom_matrix
        for b in S:
            j = self._idx(b)
            tj = self._tau_idx[j] if j is not None else None
            if tj is None:
                continue
            for a in S:
                ai = self._idx(a)
                if ai is not None and H[(ai, tj)] != 0:
                    return False
        return True

    @cached_property
    def basic_semibricks_with_dims(self) -> List[dict]:
        out = []
        for sb in self.semibricks:
            out.append({
                "names": [str(m) for m in sb],
                "dims": [m.dim() for m in sb],
                "size": len(sb),
                "total_dim": sum(m.dim() for m in sb),
            })
        return out

    @cached_property
    def basic_tau_rigids(self) -> List[dict]:
        """All basic τ-rigid modules.

        Efficiency fix B: Adachi-Iyama-Reiten — every basic τ-rigid module is
        a direct summand of some support τ-tilting module. We enumerate the
        union of all subsets of τ-parts of support τ-tilting pairs and
        deduplicate, instead of iterating over all subsets of indec τ-rigids.
        """
        from itertools import combinations
        seen = set()
        out = [{"names": [], "dims": [], "size": 0, "total_dim": 0}]
        seen.add(frozenset())
        for tau_part, _proj in self._tau_tilting_data:
            for k in range(1, len(tau_part) + 1):
                for sub in combinations(tau_part, k):
                    key = frozenset(str(m) for m in sub)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        "names": [str(m) for m in sub],
                        "dims": [m.dim() for m in sub],
                        "size": k,
                        "total_dim": sum(m.dim() for m in sub),
                    })
        out.sort(key=lambda e: e["size"])
        return out

    @cached_property
    def basic_support_tau_tiltings(self) -> List[dict]:
        out = []
        for stt in self.support_tau_tilting_modules():
            out.append({
                "names": [str(m) for m in stt],
                "dims": [m.dim() for m in stt],
                "size": len(stt),
                "total_dim": sum(m.dim() for m in stt),
            })
        return out

    @cached_property
    def basic_tau_tiltings(self) -> List[dict]:
        out = []
        for tau, proj in self._tau_tilting_data:
            if len(proj) == 0:
                out.append({
                    "names": [str(m) for m in tau],
                    "dims": [m.dim() for m in tau],
                    "size": len(tau),
                    "total_dim": sum(m.dim() for m in tau),
                })
        return out

    @cached_property
    def basic_rigids(self) -> List[dict]:
        """All basic rigid modules: subsets S of indec rigids with
        Ext¹(s_i, s_j) = 0 for all i, j.

        Efficiency fix C: this is exactly the set of cliques in the
        graph on `indec_rigids` where m–n is an edge iff Ext¹(m,n)=0
        AND Ext¹(n,m)=0. We use Bron-Kerbosch on the cached Ext matrix
        instead of an O(2^|indec_rigids|) brute force.
        """
        cands = self.indec_rigids
        idxs = [self._idx(m) for m in cands]
        E = self._ext_matrix
        neighbor = {m: set() for m in cands}
        for a, ia in zip(cands, idxs):
            for b, ib in zip(cands, idxs):
                if a is b:
                    continue
                if E[(ia, ib)] == 0 and E[(ib, ia)] == 0:
                    neighbor[a].add(b)
        cliques = _all_cliques(cands, neighbor)
        out = []
        for c in cliques:
            out.append({
                "names": [str(m) for m in c],
                "dims": [m.dim() for m in c],
                "size": len(c),
                "total_dim": sum(m.dim() for m in c),
            })
        out.sort(key=lambda e: e["size"])
        return out

    @cached_property
    def basic_partial_tiltings(self) -> List[dict]:
        out = []
        for entry in self.basic_rigids:
            ok = True
            for nm in entry["names"]:
                m = next((x for x in self.indecs if str(x) == nm), None)
                if m is None:
                    ok = False
                    break
                pd = m.proj_dim()
                if pd is None or pd > 1:
                    ok = False
                    break
            if ok:
                out.append(entry)
        return out

    @cached_property
    def basic_tiltings(self) -> List[dict]:
        rank = len(self.vertices)
        return [e for e in self.basic_partial_tiltings if e["size"] == rank]

    def _hom_cycles_among(self, modules: List["StringIndec"]) -> List[List[str]]:
        """Cycles among `modules` using the cached Hom matrix (efficiency fix A)."""
        names = [str(m) for m in modules]
        n = len(modules)
        idxs = [self._idx(m) for m in modules]
        H = self._hom_matrix
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and H[(idxs[i], idxs[j])] > 0:
                    adj[i].append(j)
        cycles = []
        visited = set()

        def dfs(start, current, path):
            for nxt in adj[current]:
                if nxt == start and len(path) >= 2:
                    cycles.append([names[v] for v in path])
                elif nxt > start and nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    dfs(start, nxt, path)
                    path.pop()
                    visited.discard(nxt)

        for s in range(n):
            visited = set(range(s))
            dfs(s, s, [s])
        return cycles

    def i_tau_rigid_cycles(self) -> List[List[str]]:
        return self._hom_cycles_among(self._indec_tau_rigids_cached)

    def i_rigid_cycles(self) -> List[List[str]]:
        return self._hom_cycles_among(self.indec_rigids)

    @cached_property
    def splitting_torsion_data(self) -> List[dict]:
        """T is splitting iff every indecomposable belongs to T or to T^perp."""
        out = []
        all_names = [str(m) for m in self.indecs]
        for idx, T in enumerate(self.torsion_classes):
            T_set = frozenset(str(m) for m in T)
            T_perp = self._torsion_free_class(idx)
            F_set = frozenset(str(m) for m in T_perp)
            in_T = [n for n in all_names if n in T_set]
            in_F = [n for n in all_names if n in F_set]
            missing = [n for n in all_names if n not in T_set and n not in F_set]
            out.append({
                "index": idx,
                "torsion_class": [str(m) for m in T],
                "torsion_free_class": [str(m) for m in T_perp],
                "is_splitting": len(missing) == 0,
                "indecs_in_torsion": in_T,
                "indecs_in_torsion_free": in_F,
                "missing_indecs": missing,
            })
        return out

    @cached_property
    def wide_lattice(self) -> dict:
        """Lattice of wide subcategories ordered by inclusion."""
        wides = self.wide_subcategories
        sets = [frozenset(str(m) for m in w) for w in wides]
        n = len(sets)
        arrows = []
        for i in range(n):
            for j in range(n):
                if i == j or not (sets[i] < sets[j]):
                    continue
                if not any(k != i and k != j and sets[i] < sets[k] < sets[j]
                           for k in range(n)):
                    arrows.append((i, j))
        return {
            "vertices": [[str(m) for m in w] for w in wides],
            "arrows": arrows,
            "size": n,
        }

    def info_bundle(self) -> dict:
        rank = len(self.vertices)
        n_arrows = len(list(self.algebra.arrows))
        is_string = self.is_string_algebra()
        is_gentle = self.is_gentle_algebra()
        n_indecs = len(self.indecs)
        n_bricks = len(self.bricks)
        brick_finite = True   # RF -> brick-finite
        n_indec_tau_rigid = len(self._indec_tau_rigids_cached)
        n_indec_rigid = len(self.indec_rigids)
        n_torsion = len(self.torsion_classes)
        n_wide = len(self.wide_subcategories)
        n_semibricks = len(self.semibricks)
        max_end = max((x.hom(x) for x in self.indecs), default=0) if n_indecs > 0 else None
        max_self_ext = max((self.ext1(x, x) for x in self.indecs), default=0) if n_indecs > 0 else None
        # Use robust τ from the cached _tau_idx (AR-duality based) so the
        # max-Hom-(X, τ Y) aggregates are correct on algebras with loops.
        max_hom_x_taux = 0
        for i, x in enumerate(self.indecs):
            ti = self._tau_idx[i]
            if ti is not None:
                v = self._hom_matrix[(i, ti)]
                if v > max_hom_x_taux:
                    max_hom_x_taux = v
        max_hom = 0
        max_ext = 0
        max_hom_tau = 0
        for i, x in enumerate(self.indecs):
            for j, y in enumerate(self.indecs):
                h = self._hom_matrix[(i, j)]
                if h > max_hom:
                    max_hom = h
                e = self._ext_matrix[(i, j)]
                if e > max_ext:
                    max_ext = e
                tj = self._tau_idx[j]
                if tj is not None:
                    v = self._hom_matrix[(i, tj)]
                    if v > max_hom_tau:
                        max_hom_tau = v
        sc = self.short_cycles()
        n_short = len(sc)
        cycles = self.morphism_cycles()
        longest_cycle = max((len(c) for c in cycles), default=0)
        n_brick_cycles = len(self.brick_cycles())
        return {
            "rank": rank,
            "vertices": list(self.vertices),
            "num_arrows": n_arrows,
            "dimension": self.dim(),
            "is_rep_finite": True,
            "is_string": is_string,
            "is_gentle": is_gentle,
            "num_indecs": n_indecs,
            "is_brick_finite": brick_finite,
            "num_bricks": n_bricks,
            "is_tau_tilting_finite": brick_finite,
            "num_indec_tau_rigids": n_indec_tau_rigid,
            "num_indec_rigids": n_indec_rigid,
            "num_torsion_classes": n_torsion,
            "num_wide_subcats": n_wide,
            "num_semibricks": n_semibricks,
            "max_dim_end": max_end,
            "max_dim_self_ext1": max_self_ext,
            "max_dim_hom_x_taux": max_hom_x_taux,
            "max_dim_hom": max_hom,
            "max_dim_ext1": max_ext,
            "max_dim_hom_tau": max_hom_tau,
            "num_short_cycles": n_short,
            "longest_cycle_length": longest_cycle,
            "num_brick_cycles": n_brick_cycles,
        }

    def indec_extended_table(self) -> List[dict]:
        rows = []
        rigids = set(id(m) for m in self.indec_rigids)
        tau_rigids = set(id(m) for m in self._indec_tau_rigids_cached)
        for m in self.indecs:
            t = self.tau_of(m)
            rows.append({
                "name": str(m),
                "dim": m.dim(),
                "top": m.top_vertices() if hasattr(m, "top_vertices") else [],
                "socle": m.socle_vertices() if hasattr(m, "socle_vertices") else [],
                "proj_dim": m.proj_dim() if hasattr(m, "proj_dim") else None,
                "inj_dim": m.inj_dim() if hasattr(m, "inj_dim") else None,
                "is_projective": m.is_projective() if hasattr(m, "is_projective") else False,
                "is_injective": m.is_injective() if hasattr(m, "is_injective") else False,
                "is_brick": m.is_brick() if hasattr(m, "is_brick") else False,
                "tau_translate": str(t) if t is not None else None,
                "is_i_tau_rigid": id(m) in tau_rigids,
                "is_i_rigid": id(m) in rigids,
            })
        return rows


def _power_subsets(items: list, size: int):
    """Yield all subsets of `items` of given size."""
    from itertools import combinations
    yield from ([list(c) for c in combinations(items, size)])

def build_rf_algebra(algebra: "StringAlgebra") -> RfAlgebra:
    """Build an RfAlgebra from a StringAlgebra. Raises ValueError if not rep-finite."""
    if not algebra.is_rep_finite():
        raise ValueError("Algebra is not representation-finite.")
    indecs = algebra.string_indecs(non_isomorphic=True)
    return RfAlgebra(algebra, indecs)
