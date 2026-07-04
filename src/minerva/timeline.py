"""Temps diégétique : moments narratifs, contraintes d'ordre, résolveur.

Le temps d'un roman n'est pas un axe numérique mais un graphe de contraintes
qualitatives (avant / simultané / pendant) portant des écarts quantifiés quand
le texte les donne. Le résolveur (code pur, jamais le LLM) en dérive un ordre
total stable et, quand un chemin d'écarts le permet, une coordonnée en jours —
dérivés recalculables à volonté, jamais une précision inventée.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

AVANT = "avant"
SIMULTANE = "simultané"
PENDANT = "pendant"
RELATIONS = frozenset({AVANT, SIMULTANE, PENDANT})

# Unités françaises -> jours (approximations assumées : mois = 30 j, an = 365 j).
_UNITS_DAYS = {
    "heure": 1 / 24, "nuit": 1.0, "jour": 1.0, "journée": 1.0,
    "semaine": 7.0, "mois": 30.0, "an": 365.0, "année": 365.0,
}


def gap_to_days(value: float | None, unit: str) -> float | None:
    """Convertit un écart quantifié en jours ; None si inconvertible."""
    if not value or value <= 0:
        return None
    key = unit.strip().casefold()
    days = _UNITS_DAYS.get(key) or _UNITS_DAYS.get(key.rstrip("s"))
    return value * days if days else None


class Gap(BaseModel):
    text: str = ""          # expression du texte (« vingt ans après »)
    days: float | None = None  # normalisation en jours, si quantifiable


class Moment(BaseModel):
    id: int
    chunk_index: int
    seq: int                # position du moment dans son chunk
    summary: str = ""
    resolved_order: int | None = None   # dérivé (résolveur), recalculable
    resolved_days: float | None = None  # dérivé (résolveur), recalculable

    def reading_key(self) -> tuple[int, int]:
        return (self.chunk_index, self.seq)


class TemporalConstraint(BaseModel):
    source_id: int
    relation: str           # AVANT : source précède target ; SIMULTANE/PENDANT
    target_id: int
    gap: Gap = Field(default_factory=Gap)


class Timeline:
    """Conteneur des moments, contraintes et présences d'entités."""

    def __init__(self) -> None:
        self._moments: dict[int, Moment] = {}
        self._constraints: list[TemporalConstraint] = []
        self._appearances: dict[int, set[str]] = {}  # moment_id -> noms canoniques
        self._next_id = 1

    @property
    def moments(self) -> list[Moment]:
        return sorted(self._moments.values(), key=lambda m: m.id)

    @property
    def constraints(self) -> list[TemporalConstraint]:
        return list(self._constraints)

    @property
    def appearances(self) -> dict[int, set[str]]:
        return {k: set(v) for k, v in self._appearances.items()}

    def moment(self, moment_id: int) -> Moment | None:
        return self._moments.get(moment_id)

    def add_moment(self, chunk_index: int, seq: int, summary: str = "") -> Moment:
        moment = Moment(
            id=self._next_id, chunk_index=chunk_index, seq=seq, summary=summary.strip()
        )
        self._moments[moment.id] = moment
        self._next_id += 1
        return moment

    def load_moment(self, moment: Moment) -> None:
        """Restaure un moment persisté (id imposé) — usage store.load."""
        self._moments[moment.id] = moment
        self._next_id = max(self._next_id, moment.id + 1)

    def add_constraint(
        self, source_id: int, relation: str, target_id: int, gap: Gap | None = None
    ) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"Relation temporelle inconnue : {relation!r}")
        if source_id not in self._moments or target_id not in self._moments:
            raise ValueError(f"Contrainte sur moment inexistant : {source_id} -> {target_id}")
        self._constraints.append(
            TemporalConstraint(
                source_id=source_id, relation=relation, target_id=target_id,
                gap=gap or Gap(),
            )
        )

    def add_appearance(self, moment_id: int, entity_name: str) -> None:
        if moment_id in self._moments and entity_name.strip():
            self._appearances.setdefault(moment_id, set()).add(entity_name.strip())

    def recent(self, n: int) -> list[Moment]:
        """Les n derniers moments en ordre de lecture (pour le prompt)."""
        ordered = sorted(self._moments.values(), key=Moment.reading_key)
        return ordered[-n:]

    def rename_entity(self, old: str, new: str) -> None:
        for names in self._appearances.values():
            if old in names:
                names.discard(old)
                names.add(new)

    def clone(self) -> Timeline:
        copy = Timeline()
        for m in self.moments:
            copy.load_moment(m.model_copy())
        copy._constraints = [c.model_copy(deep=True) for c in self._constraints]
        copy._appearances = {k: set(v) for k, v in self._appearances.items()}
        return copy
