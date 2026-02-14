class ProductionFunction:
    """Simple linear production: output = productivity_factor * num_workers."""

    def __init__(self, config):
        self.productivity = config.productivity_factor

    def produce(self, firm) -> float:
        """Run production for a firm. Adds output to inventory, returns units produced."""
        output = self.productivity * len(firm.employees)
        firm.inventory += output
        return output
