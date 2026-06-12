from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Chemical:
    id: str
    name: str
    layer: int
    molecular_weight: float
    melting_point: float
    boiling_point: float
    base_toxicity: float
    medicinal_expected: float
    medicinal_efficacy: float
    price_per_gram: Optional[float] = None
    heat_capacity_J_per_gK: float = 2.0
    latent_heat_fusion_J_per_g: float = 150.0
    latent_heat_vaporization_J_per_g: float = 800.0
    clausius_C: float = 40.0
    is_solvent: bool = False
    solubility: Dict[str, float] = field(default_factory=dict)

    @property
    def medicinal_value(self) -> float:
        return self.medicinal_expected * self.medicinal_efficacy

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "layer": self.layer,
            "molecular_weight": round(self.molecular_weight, 2),
            "melting_point": round(self.melting_point, 2),
            "boiling_point": round(self.boiling_point, 2),
            "base_toxicity": round(self.base_toxicity, 3),
            "medicinal_expected": round(self.medicinal_expected, 3),
            "medicinal_efficacy": round(self.medicinal_efficacy, 4),
            "price_per_gram": round(self.price_per_gram, 4) if self.price_per_gram is not None else None,
            "heat_capacity_J_per_gK": round(self.heat_capacity_J_per_gK, 4),
            "latent_heat_fusion_J_per_g": round(self.latent_heat_fusion_J_per_g, 2),
            "latent_heat_vaporization_J_per_g": round(self.latent_heat_vaporization_J_per_g, 2),
            "clausius_C": round(self.clausius_C, 2),
            "is_solvent": self.is_solvent,
        }
        if self.solubility:
            d["solubility"] = {k: round(v, 2) for k, v in self.solubility.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Chemical:
        return cls(
            id=data["id"],
            name=data["name"],
            layer=data["layer"],
            molecular_weight=data["molecular_weight"],
            melting_point=data["melting_point"],
            boiling_point=data["boiling_point"],
            base_toxicity=data["base_toxicity"],
            medicinal_expected=data["medicinal_expected"],
            medicinal_efficacy=data["medicinal_efficacy"],
            price_per_gram=data.get("price_per_gram"),
            heat_capacity_J_per_gK=data.get("heat_capacity_J_per_gK", 2.0),
            latent_heat_fusion_J_per_g=data.get("latent_heat_fusion_J_per_g", 150.0),
            latent_heat_vaporization_J_per_g=data.get("latent_heat_vaporization_J_per_g", 800.0),
            clausius_C=data.get("clausius_C", 40.0),
            is_solvent=data.get("is_solvent", False),
            solubility=data.get("solubility", {}),
        )


EQUIPMENT_CATALOG: Dict[str, Dict] = {
    "open_beaker": {
        "description": "Open beaker at atmospheric pressure, exchanges heat with environment",
        "vessel_type": "open",
        "thermal_mode": "open_air",
        "max_pressure_atm": 1.0,
        "max_temp_C": 300.0,
        "min_temp_C": -20.0,
        "max_capacity_g": 500.0,
        "base_cost_per_hour": 2.0,
        "cost_multiplier": 1.0,
        "heat_transfer_coeff": 0.05,
    },
    "reflux_condenser": {
        "description": "Round-bottom flask with reflux condenser, moderate insulation, constant pressure",
        "vessel_type": "open",
        "thermal_mode": "open_air",
        "max_pressure_atm": 1.5,
        "max_temp_C": 400.0,
        "min_temp_C": -20.0,
        "max_capacity_g": 1000.0,
        "base_cost_per_hour": 5.0,
        "cost_multiplier": 1.5,
        "heat_transfer_coeff": 0.02,
    },
    "sealed_flask": {
        "description": "Sealed flask, constant volume, partially insulated",
        "vessel_type": "sealed",
        "thermal_mode": "adiabatic",
        "max_pressure_atm": 5.0,
        "max_temp_C": 400.0,
        "min_temp_C": -40.0,
        "max_capacity_g": 500.0,
        "base_cost_per_hour": 8.0,
        "cost_multiplier": 2.0,
    },
    "autoclave": {
        "description": "High-pressure sealed reactor, temperature-controlled",
        "vessel_type": "sealed",
        "thermal_mode": "isothermal",
        "max_pressure_atm": 50.0,
        "max_temp_C": 600.0,
        "min_temp_C": -60.0,
        "max_capacity_g": 2000.0,
        "base_cost_per_hour": 20.0,
        "cost_multiplier": 4.0,
        "max_heat_rate_W": 500.0,
    },
    "insulated_reactor": {
        "description": "Well-insulated sealed reactor, adiabatic conditions",
        "vessel_type": "sealed",
        "thermal_mode": "adiabatic",
        "max_pressure_atm": 20.0,
        "max_temp_C": 800.0,
        "min_temp_C": -80.0,
        "max_capacity_g": 1500.0,
        "base_cost_per_hour": 15.0,
        "cost_multiplier": 3.0,
    },
    "heated_reactor": {
        "description": "Sealed reactor with continuous heating element",
        "vessel_type": "sealed",
        "thermal_mode": "heating",
        "max_pressure_atm": 20.0,
        "max_temp_C": 1000.0,
        "min_temp_C": -20.0,
        "max_capacity_g": 1500.0,
        "base_cost_per_hour": 25.0,
        "cost_multiplier": 4.5,
    },
    "cooled_reactor": {
        "description": "Sealed reactor with active cooling system",
        "vessel_type": "sealed",
        "thermal_mode": "cooling",
        "max_pressure_atm": 20.0,
        "max_temp_C": 400.0,
        "min_temp_C": -196.0,
        "max_capacity_g": 1500.0,
        "base_cost_per_hour": 30.0,
        "cost_multiplier": 5.0,
    },
}


@dataclass
class Reaction:
    id: str
    reactants: List[Tuple[str, int]]
    catalysts: List[str]
    products: List[Tuple[str, int]]
    byproducts: List[Tuple[str, int]]
    delta_G_kJ: float
    delta_H_kJ: float
    activation_energy_kJ: float
    log_A_factor: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "reactants": [[cid, coeff] for cid, coeff in self.reactants],
            "catalysts": list(self.catalysts),
            "products": [[cid, coeff] for cid, coeff in self.products],
            "byproducts": [[cid, coeff] for cid, coeff in self.byproducts],
            "delta_G_kJ": round(self.delta_G_kJ, 3),
            "delta_H_kJ": round(self.delta_H_kJ, 3),
            "activation_energy_kJ": round(self.activation_energy_kJ, 3),
            "log_A_factor": round(self.log_A_factor, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Reaction:
        return cls(
            id=data["id"],
            reactants=[tuple(x) for x in data["reactants"]],
            catalysts=list(data["catalysts"]),
            products=[tuple(x) for x in data["products"]],
            byproducts=[tuple(x) for x in data["byproducts"]],
            delta_G_kJ=data["delta_G_kJ"],
            delta_H_kJ=data["delta_H_kJ"],
            activation_energy_kJ=data["activation_energy_kJ"],
            log_A_factor=data["log_A_factor"],
        )


DEFAULT_COST_PARAMS: Dict[str, float] = {
    "heating_coeff": 0.8,
    "cooling_coeff": 1.2,
    "heating_exponent": 1.5,
    "cooling_exponent": 1.3,
    "pressure_high_coeff": 1.5,
    "pressure_low_coeff": 1.5,
    "pressure_high_exp": 0.7,
    "pressure_low_exp": 0.6,
    "equipment_base": 5.0,
    "equipment_pressure_coeff": 0.3,
    "duration_coeff": 0.05,
}


@dataclass
class World:
    world_id: str
    seed: int
    chemicals: Dict[str, Chemical] = field(default_factory=dict)
    reactions: Dict[str, Reaction] = field(default_factory=dict)
    cost_params: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COST_PARAMS))
    equipment: Dict[str, Dict] = field(default_factory=lambda: dict(EQUIPMENT_CATALOG))

    @property
    def num_layers(self) -> int:
        if not self.chemicals:
            return 0
        return max(c.layer for c in self.chemicals.values())

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "metadata": {
                "num_layers": self.num_layers,
                "seed": self.seed,
                "num_chemicals": len(self.chemicals),
                "num_reactions": len(self.reactions),
            },
            "chemicals": {cid: c.to_dict() for cid, c in self.chemicals.items()},
            "reactions": {rid: r.to_dict() for rid, r in self.reactions.items()},
            "cost_params": {k: round(v, 4) for k, v in self.cost_params.items()},
            "equipment": self.equipment,
        }

    @classmethod
    def from_dict(cls, data: dict) -> World:
        world = cls(
            world_id=data["world_id"],
            seed=data["metadata"]["seed"],
        )
        world.chemicals = {cid: Chemical.from_dict(cdata) for cid, cdata in data["chemicals"].items()}
        world.reactions = {rid: Reaction.from_dict(rdata) for rid, rdata in data["reactions"].items()}
        saved_params = data.get("cost_params", {})
        world.cost_params = {**DEFAULT_COST_PARAMS, **saved_params}
        world.equipment = data.get("equipment", dict(EQUIPMENT_CATALOG))
        return world

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> World:
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
