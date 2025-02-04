# mineral_groups.py
from typing import Dict, Optional

class MineralHierarchy:
    """Represents the leaf-level mineral compositions with their parent group weights"""
    def __init__(self):
        # Common minerals shared between rock types
        self.shared_minerals = {
            # Feldspar minerals
            'orthoclase': {
                'group': 'alkali_feldspars',
                'parent_group': 'feldspars'
            },
            'albite': {
                'group': 'alkali_feldspars',
                'parent_group': 'feldspars'
            },
            'anorthite': {
                'group': 'plagioclase',
                'parent_group': 'feldspars'
            },
            # Mica minerals
            'annite': {
                'group': 'mica_group',
                'parent_group': 'micas'
            },
            'phlogopite': {
                'group': 'mica_group',
                'parent_group': 'micas'
            },
            'muscovite': {
                'group': 'mica_group',
                'parent_group': 'micas'
            }
        }

        self.granite_minerals = {
            **self.shared_minerals,
            'quartz': {
                'group': 'quartz',
                'parent_group': 'quartz',
                'parent_weight': 0.6,  # 20-40%
            }
        }

        self.sandstone_minerals = {
            **self.shared_minerals,
            'quartz': {
                'group': 'quartz',
                'parent_group': 'quartz',
                'parent_weight': 0.9,  # >70%
            },
            'calcite': {
                'group': 'calcite',
                'parent_group': 'calcite',
                'parent_weight': 0.2,  # <10%
            },
            'pyrite': {
                'group': 'pyrite',
                'parent_group': 'pyrite',
                'parent_weight': 0.01,  # <1%
            },
            'rutile': {
                'group': 'rutile',
                'parent_group': 'rutile',
                'parent_weight': 0.02,  # <2%
            },
            'tourmaline': {
                'group': 'tourmaline',
                'parent_group': 'tourmaline',
                'parent_weight': 0.02,  # <2%
            }
        }

        self.limestone_minerals = {
            'calcite': {
                'group': 'calcite',
                'parent_group': 'pure_limestone',
                'parent_weight': 0.95,  # >90% for pure limestone
            },
            'dolomite': {
                'group': 'dolomite',
                'parent_group': 'dolomitic_limestone',
                'parent_weight': 0.3,  # 10-50% for dolomitic limestone
            },
            'quartz': {
                'group': 'accessory',
                'parent_group': 'accessory_minerals',
                'parent_weight': 0.1,  # <10%
            },
            'pyrite': {
                'group': 'accessory',
                'parent_group': 'accessory_minerals',
                'parent_weight': 0.05,  # <5%
            },
            'orthoclase': {
                'group': 'alkali_feldspars',
                'parent_group': 'accessory_minerals',
                'parent_weight': 0.05,  # <5% total feldspars
            },
            'albite': {
                'group': 'alkali_feldspars',
                'parent_group': 'accessory_minerals',
                'parent_weight': 0.05
            },
            'anorthite': {
                'group': 'plagioclase',
                'parent_group': 'accessory_minerals',
                'parent_weight': 0.05
            }
        }

        # Define composition constraints for each rock type
        self.composition_constraints = {
            'granite': {
                'feldspars': (0.45, 0.80),
                'quartz': (0.20, 0.40),
                'micas': (0.0, 0.15)
            },
            'sandstone': {
                'quartz': (0.70, 1.0),
                'feldspars': (0.05, 0.25),
                'calcite': (0.0, 0.10),
                'pyrite': (0.0, 0.01),
                'mica_group': (0.02, 0.03),
                'rutile': (0.0, 0.02),
                'tourmaline': (0.0, 0.02)
            },
            'limestone': {
                'pure': {
                    'calcite': (0.90, 1.0)
                },
                'dolomitic': {
                    'calcite': (0.50, 0.90),
                    'dolomite': (0.10, 0.50),
                    'quartz': (0.0, 0.10)
                },
                'accessory': {
                    'feldspars': (0.0, 0.05),
                    'pyrite': (0.0, 0.05)
                }
            }
        }

    def get_mineral_details(self, mineral_name: str, rock_type: str) -> Dict:
        """Get mineral details for a specific rock type"""
        if rock_type == 'granite':
            return self.granite_minerals.get(mineral_name.lower(), {})
        elif rock_type == 'sandstone':
            return self.sandstone_minerals.get(mineral_name.lower(), {})
        elif rock_type == 'limestone':
            return self.limestone_minerals.get(mineral_name.lower(), {})
        return {}

    def get_composition_constraints(self, rock_type: str) -> Dict:
        """Get composition constraints for a specific rock type"""
        return self.composition_constraints.get(rock_type, {})