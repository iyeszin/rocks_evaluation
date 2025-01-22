from dataclasses import dataclass

@dataclass
class MineralGroups:
    """Mineral groups definition following Γ notation"""
    def __init__(self):
        self.feldspars = {'albite', 'anorthite', 'orthoclase', 'sanidine'}
        self.quartz = {'quartz'}
        self.micas = {'annite', 'eastonite', 'margarite', 'muscovite', 'phlogopite'}
        self.calcite = {'calcite'}
        self.pyrite = {'pyrite'}
        self.rutile = {'rutile'}
        self.tourmaline = {'tourmaline'}
