from econ_sim.config import Industry


class ProductionFunction:
    """Computes output for each firm based on workforce and input availability."""

    def __init__(self, config):
        self.config = config

    def produce(self, firm, worker_map: dict) -> float:
        """Run production for one firm. Returns units produced and adds to inventory.

        For industries with upstream inputs (Housing, Technology), output is
        constrained by both workforce capacity and available input materials.
        Workers with unmet needs have a reduced productivity modifier.
        """
        ind_config = self.config.industry_configs[firm.industry]

        if not firm.employees:
            return 0.0

        # Effective workforce accounting for productivity modifiers
        # (workers with unmet food/shelter are less productive)
        total_effective_labor = 0.0
        for emp_name in firm.employees:
            worker = worker_map.get(emp_name)
            modifier = getattr(worker, "productivity_modifier", 1.0) if worker else 1.0
            skill = getattr(worker, "skill_level", 1.0) if worker else 1.0
            total_effective_labor += skill * modifier

        # Base output from labor
        workforce_capacity = ind_config.productivity_factor * total_effective_labor

        # Apply input constraint for upstream-dependent industries
        if ind_config.input_industry is not None and ind_config.input_ratio > 0:
            # Maximum output supportable by current input inventory
            input_capacity = firm.input_inventory / ind_config.input_ratio if ind_config.input_ratio > 0 else float("inf")
            effective_output = min(workforce_capacity, input_capacity)

            # Consume inputs proportional to output produced
            inputs_consumed = effective_output * ind_config.input_ratio
            firm.input_inventory = max(0.0, firm.input_inventory - inputs_consumed)
            firm.costs += 0.0  # input purchase cost was already deducted in input market phase
        else:
            effective_output = workforce_capacity

        firm.inventory += effective_output
        return effective_output
