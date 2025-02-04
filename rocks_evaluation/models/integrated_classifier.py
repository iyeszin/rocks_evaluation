import torch
import numpy as np
from typing import Dict, List, Tuple

from ..utils.enums import RockType
from ..utils.data_classes import MineralGroups
from ..utils.visualization import plot_mineral_analysis

from .mineral_groups import MineralHierarchy

class IntegratedRockClassifier:
    """Base classifier for rock classification without uncertainty handling"""
    def __init__(self, model, label_encoder, device, window_size: int = 10):
        self.model = model
        self.label_encoder = label_encoder
        self.device = device
        self.window_size = window_size
        self.mineral_groups = MineralGroups()
        self.prediction_history = []
        self.ground_truth_history = []
        self.analysis_history = []

    def predict_mineral(self, spectrum: torch.Tensor) -> Tuple[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            if len(spectrum.shape) == 1:
                spectrum = spectrum.unsqueeze(0)
            spectrum = spectrum.to(self.device)
            outputs = self.model(spectrum)
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
            predicted_idx = np.argmax(probs)
            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            return predicted_label, probs

    def check_accuracy_rule(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        if len(predictions) != 10 or len(ground_truth) != 10:
            return {'satisfied': False, 'accuracy': 0.0, 'correct_predictions': 0}
        
        correct_predictions = sum(1 for pred, truth in zip(predictions, ground_truth)
                                if pred.lower() == truth.lower())
        accuracy = correct_predictions / 10
        
        return {
            'satisfied': accuracy >= 0.6,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions
        }

    def check_mineral_assemblage_rules(self, predictions: List[str]) -> Dict:
        predictions_lower = {pred.lower() for pred in predictions}
        
        granite_rules = {
            'feldspar_present': bool(predictions_lower & self.mineral_groups.feldspars),
            'quartz_present': bool(predictions_lower & self.mineral_groups.quartz),
            'mica_present': bool(predictions_lower & self.mineral_groups.micas)
        }
        
        limestone_rules = {
            'calcite_present': bool(predictions_lower & self.mineral_groups.calcite),
            'low_impurities': len(predictions_lower & (self.mineral_groups.quartz | 
                                                    self.mineral_groups.feldspars | 
                                                    self.mineral_groups.pyrite)) <= 2
        }
        
        sandstone_rules = {
            'quartz_present': bool(predictions_lower & self.mineral_groups.quartz),
            'feldspar_present': bool(predictions_lower & self.mineral_groups.feldspars),
            'accessory_present': bool(predictions_lower & (self.mineral_groups.micas | 
                                                       self.mineral_groups.rutile | 
                                                       self.mineral_groups.tourmaline))
        }
        
        mineral_counts = {
            'feldspars': sum(1 for p in predictions if p.lower() in self.mineral_groups.feldspars),
            'quartz': sum(1 for p in predictions if p.lower() in self.mineral_groups.quartz),
            'micas': sum(1 for p in predictions if p.lower() in self.mineral_groups.micas),
            'calcite': sum(1 for p in predictions if p.lower() in self.mineral_groups.calcite),
            'pyrite': sum(1 for p in predictions if p.lower() in self.mineral_groups.pyrite),
            'rutile': sum(1 for p in predictions if p.lower() in self.mineral_groups.rutile),
            'tourmaline': sum(1 for p in predictions if p.lower() in self.mineral_groups.tourmaline)
        }
        
        is_granite = all(granite_rules.values())
        is_limestone = all(limestone_rules.values())
        is_sandstone = all(sandstone_rules.values())
        
        return {
            'satisfied': any([is_granite, is_limestone, is_sandstone]),
            'details': {
                'granite': granite_rules,
                'limestone': limestone_rules,
                'sandstone': sandstone_rules
            },
            'counts': mineral_counts,
            'rock_types': {
                'granite': is_granite,
                'limestone': is_limestone,
                'sandstone': is_sandstone
            }
        }

    def process_spectrum(self, spectrum: torch.Tensor, true_mineral: str = None) -> Dict:
        predicted_mineral, probabilities = self.predict_mineral(spectrum)
        
        self.prediction_history.append(predicted_mineral)
        if true_mineral is not None:
            self.ground_truth_history.append(true_mineral)
        
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
            if self.ground_truth_history:
                self.ground_truth_history.pop(0)
        
        if len(self.prediction_history) == 10:
            accuracy_analysis = self.check_accuracy_rule(
                self.prediction_history, 
                self.ground_truth_history if self.ground_truth_history else self.prediction_history
            )
            assemblage_analysis = self.check_mineral_assemblage_rules(self.prediction_history)
            
            if accuracy_analysis['satisfied']:
                if assemblage_analysis['rock_types']['granite']:
                    classification = RockType.GRANITE
                elif assemblage_analysis['rock_types']['limestone']:
                    classification = RockType.LIMESTONE
                elif assemblage_analysis['rock_types']['sandstone']:
                    classification = RockType.SANDSTONE
                else:
                    classification = RockType.OTHER
            else:
                classification = RockType.OTHER
            
            rock_analysis = {
                'classification': classification.value,
                'accuracy_rule': accuracy_analysis,
                'assemblage_rules': assemblage_analysis,
            }
        else:
            rock_analysis = {
                'classification': RockType.OTHER.value,
                'status': 'Insufficient measurements',
                'current_count': len(self.prediction_history)
            }
        
        self.analysis_history.append({
            'mineral_prediction': predicted_mineral,
            'true_mineral': true_mineral,
            'rock_analysis': rock_analysis,
            'measurement_number': len(self.prediction_history)
        })
        
        return {
            'mineral_prediction': predicted_mineral,
            'mineral_probabilities': probabilities,
            'rock_analysis': rock_analysis
        }

    def plot_analysis(self, rock_num, save_path=None):
        plot_mineral_analysis(self.analysis_history, rock_num, save_path)


class UncertaintyIntegratedRockClassifier(IntegratedRockClassifier):
    """Extends IntegratedRockClassifier with uncertainty handling"""
    def __init__(self, model, label_encoder, device, 
                 entropy_threshold=1.3, 
                 variance_threshold=0.07):
        super().__init__(model, label_encoder, device)
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.uncertainty_history = []

    def predict_mineral(self, spectrum: torch.Tensor) -> Tuple[str, np.ndarray, float, float]:
        self.model.eval()
        predictions = []
        
        for _ in range(20):
            with torch.no_grad():
                if len(spectrum.shape) == 1:
                    spectrum = spectrum.unsqueeze(0)
                outputs = self.model(spectrum.to(self.device), enable_dropout=True)
                probs = torch.softmax(outputs, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)
        variance = predictions.var(dim=0).mean(dim=1)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        is_unknown = (entropy > self.entropy_threshold) & (variance > self.variance_threshold)
        is_unknown = is_unknown | (mean_probs.max() < 0.30)
        
        if is_unknown.any():
            return "unknown", mean_probs.cpu().numpy()[0], entropy.item(), variance.item()
        else:
            predicted_idx = mean_probs.argmax(dim=1).item()
            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            return predicted_label, mean_probs.cpu().numpy()[0], entropy.item(), variance.item()

    def check_accuracy_rule(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        if len(predictions) != 10 or len(ground_truth) != 10:
            return {'satisfied': False, 'accuracy': 0.0, 'correct_predictions': 0, 'unknown_count': 0}
        
        unknown_count = sum(1 for pred in predictions if pred == "unknown")
        valid_predictions = [(p, t) for p, t in zip(predictions, ground_truth) if p != "unknown"]
        
        if not valid_predictions:
            return {'satisfied': False, 'accuracy': 0.0, 'correct_predictions': 0, 'unknown_count': unknown_count}
        
        correct_predictions = sum(1 for pred, truth in valid_predictions if pred.lower() == truth.lower())
        accuracy = correct_predictions / (len(valid_predictions) if valid_predictions else 1)
        
        return {
            'satisfied': accuracy >= 0.6 and unknown_count <= 3,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'unknown_count': unknown_count
        }

    def process_spectrum(self, spectrum: torch.Tensor, true_mineral: str = None) -> Dict:
        predicted_mineral, probabilities, entropy, variance = self.predict_mineral(spectrum)
        
        self.prediction_history.append(predicted_mineral)
        self.uncertainty_history.append({'entropy': entropy, 'variance': variance})
        if true_mineral is not None:
            self.ground_truth_history.append(true_mineral)
        
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
            self.uncertainty_history.pop(0)
            if self.ground_truth_history:
                self.ground_truth_history.pop(0)
        
        if len(self.prediction_history) == 10:
            accuracy_analysis = self.check_accuracy_rule(
                self.prediction_history, 
                self.ground_truth_history if self.ground_truth_history else self.prediction_history
            )
            assemblage_analysis = self.check_mineral_assemblage_rules(self.prediction_history)
            
            if accuracy_analysis['satisfied'] and assemblage_analysis['satisfied']:
                if assemblage_analysis['rock_types']['granite']:
                    classification = RockType.GRANITE
                elif assemblage_analysis['rock_types']['limestone']:
                    classification = RockType.LIMESTONE
                elif assemblage_analysis['rock_types']['sandstone']:
                    classification = RockType.SANDSTONE
                else:
                    classification = RockType.OTHER
            else:
                classification = RockType.OTHER
            
            rock_analysis = {
                'classification': classification.value,
                'accuracy_rule': accuracy_analysis,
                'assemblage_rules': assemblage_analysis,
                'uncertainty_metrics': {
                    'mean_entropy': np.mean([u['entropy'] for u in self.uncertainty_history]),
                    'mean_variance': np.mean([u['variance'] for u in self.uncertainty_history]),
                    'unknown_predictions': accuracy_analysis['unknown_count']
                }
            }
        else:
            rock_analysis = {
                'classification': RockType.OTHER.value,
                'status': 'Insufficient measurements',
                'current_count': len(self.prediction_history)
            }
        
        self.analysis_history.append({
            'mineral_prediction': predicted_mineral,
            'true_mineral': true_mineral,
            'uncertainty': {'entropy': entropy, 'variance': variance},
            'rock_analysis': rock_analysis,
            'measurement_number': len(self.prediction_history)
        })
        
        return {
            'mineral_prediction': predicted_mineral,
            'mineral_probabilities': probabilities,
            'uncertainty': {'entropy': entropy, 'variance': variance},
            'rock_analysis': rock_analysis
        }
    
class HierarchicalRockClassifier(IntegratedRockClassifier):
    """Adds hierarchical mineral structure to base classifier"""
    def __init__(self, model, label_encoder, device, window_size: int = 10):
        super().__init__(model, label_encoder, device, window_size)
        self.hierarchy = MineralHierarchy()

    def get_mineral_counts(self, predictions: List[str]) -> Dict[str, int]:
        """Count occurrences of each mineral in predictions"""
        predictions_lower = [p.lower() for p in predictions]
        return {
            'quartz': predictions_lower.count('quartz'),
            'feldspars': (
                predictions_lower.count('orthoclase') +
                predictions_lower.count('albite') +
                predictions_lower.count('anorthite')
            ),
            'micas': (
                predictions_lower.count('annite') +
                predictions_lower.count('phlogopite') +
                predictions_lower.count('muscovite')
            ),
            'calcite': predictions_lower.count('calcite'),
            'pyrite': predictions_lower.count('pyrite'),
            'rutile': predictions_lower.count('rutile'),
            'tourmaline': predictions_lower.count('tourmaline')
        }

    def evaluate_rock_type_weights(self, predictions: List[str]) -> Dict[str, float]:
        """
        Evaluate likelihood weights for each rock type based on mineral proportions
        """
        if len(predictions) != self.window_size:
            return {'granite': 0.0, 'sandstone': 0.0, 'limestone': 0.0}
        
        # Add debug printing
        print("\nDEBUG: Predictions received:", predictions)

        counts = self.get_mineral_counts(predictions)
        total = len(predictions)

        # Print counts
        print("\nDEBUG: Mineral counts:")
        for mineral, count in counts.items():
            print(f"{mineral}: {count}")

        # Calculate key ratios
        quartz_ratio = counts['quartz'] / total
        feldspar_ratio = counts['feldspars'] / total
        mica_ratio = counts['micas'] / total
        calcite_ratio = counts['calcite'] / total
        
        print("\nDEBUG: Mineral ratios:")
        print(f"Quartz ratio: {quartz_ratio:.2f}")
        print(f"Feldspar ratio: {feldspar_ratio:.2f}")
        print(f"Mica ratio: {mica_ratio:.2f}")
        print(f"Calcite ratio: {calcite_ratio:.2f}")

        weights = {
            'granite': 0.0,
            'sandstone': 0.0,
            'limestone': 0.0
        }

        # Granite criteria check
        print("\nDEBUG: Checking granite criteria:")
        print(f"0.2 <= quartz_ratio <= 0.4: {0.2 <= quartz_ratio <= 0.4}")
        print(f"0.45 <= feldspar_ratio <= 0.8: {0.45 <= feldspar_ratio <= 0.8}")
        print(f"0.0 <= mica_ratio <= 0.15: {0.0 <= mica_ratio <= 0.15}")

         # Granite scoring - partial weights for each criterion
        granite_score = 0.0
        if 0.2 <= quartz_ratio <= 0.4:
            granite_score += 0.4  # 40% of weight for correct quartz ratio
        if 0.35 <= feldspar_ratio <= 0.8:  # Made more flexible
            granite_score += 0.4  # 40% of weight for correct feldspar ratio
        if 0.0 <= mica_ratio <= 0.35:      # Made more flexible
            granite_score += 0.2  # 20% of weight for correct mica ratio
        weights['granite'] = granite_score

        # Sandstone criteria
        sandstone_score = 0.0
        if quartz_ratio >= 0.65:  # Made more flexible
            sandstone_score += 0.5
        if 0.05 <= feldspar_ratio <= 0.25:
            sandstone_score += 0.3
        if calcite_ratio <= 0.1:
            sandstone_score += 0.2
        weights['sandstone'] = sandstone_score

        # Limestone criteria
        limestone_score = 0.0
        if calcite_ratio >= 0.85:
            limestone_score = 1.0
        elif calcite_ratio >= 0.45:
            limestone_score = 0.8
        weights['limestone'] = limestone_score

        # Normalize weights
        max_weight = max(weights.values())
        if max_weight > 0:
            weights = {k: v/max_weight for k, v in weights.items()}

        return weights

    def check_mineral_assemblage_rules(self, predictions: List[str]) -> Dict:
        """Check mineral assemblages using weight-based approach"""
        weights = self.evaluate_rock_type_weights(predictions)
        
        CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence needed
        DOMINANCE_THRESHOLD = 0.3   # Minimum difference needed between highest and second-highest weight
        
        # Get the two highest weights
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        highest_weight = sorted_weights[0][1]
        second_highest = sorted_weights[1][1] if len(sorted_weights) > 1 else 0
        
        # Check if we're confident enough
        is_confident = (highest_weight >= CONFIDENCE_THRESHOLD and 
                    (highest_weight - second_highest) >= DOMINANCE_THRESHOLD)
        
        rock_types = {
            'granite': False,
            'limestone': False,
            'sandstone': False
        }
        
        if is_confident:
            winning_rock = sorted_weights[0][0]
            rock_types[winning_rock] = True
        
        return {
            'satisfied': is_confident,
            'weights': weights,
            'rock_types': rock_types,
            'counts': self.get_mineral_counts(predictions)
        }

    def process_spectrum(self, spectrum: torch.Tensor, true_mineral: str = None) -> Dict:
        """Process spectrum and determine rock type"""
        predicted_mineral, probabilities = self.predict_mineral(spectrum)
        
        self.prediction_history.append(predicted_mineral)
        if true_mineral is not None:
            self.ground_truth_history.append(true_mineral)
        
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
            if self.ground_truth_history:
                self.ground_truth_history.pop(0)
        
        if len(self.prediction_history) == self.window_size:
            mineral_analysis = self.check_mineral_assemblage_rules(self.prediction_history)
            
            # Determine classification based on weights
            if mineral_analysis['satisfied']:
                weights = mineral_analysis['weights']
                max_rock = max(weights.items(), key=lambda x: x[1])[0].upper()
                classification = RockType[max_rock]
            else:
                classification = RockType.OTHER
                
            rock_analysis = {
                'classification': classification.value,
                'weights': mineral_analysis['weights'],
                'mineral_counts': mineral_analysis['counts'],
                'confidence': max(mineral_analysis['weights'].values()),
                'is_confident': mineral_analysis['satisfied'] 
            }
        else:
            rock_analysis = {
                'classification': RockType.OTHER.value,
                'status': 'Insufficient measurements',
                'current_count': len(self.prediction_history)
            }

        # Store in analysis_history - This is the key addition
        self.analysis_history.append({
            'mineral_prediction': predicted_mineral,
            'true_mineral': true_mineral,
            'rock_analysis': rock_analysis,
            'measurement_number': len(self.prediction_history)
        })
        
        return {
            'mineral_prediction': predicted_mineral,
            'mineral_probabilities': probabilities,
            'rock_analysis': rock_analysis
        }
    

class UncertaintyHierarchicalRockClassifier(UncertaintyIntegratedRockClassifier):
    """Adds hierarchical mineral structure to uncertainty-aware classifier"""
    def __init__(self, model, label_encoder, device, 
                 entropy_threshold=1.3, 
                 variance_threshold=0.07):
        super().__init__(model, label_encoder, device, 
                        entropy_threshold, variance_threshold)
        self.hierarchy = MineralHierarchy()

    def evaluate_rock_composition(self, 
                                predictions: List[str], 
                                uncertainties: List[Dict[str, float]], 
                                rock_type: str) -> Dict:
        """
        Evaluate rock composition considering prediction uncertainties
        
        Args:
            predictions: List of predicted minerals
            uncertainties: List of dicts containing 'entropy' and 'variance' for each prediction
            rock_type: Type of rock to evaluate against ('granite', 'sandstone', 'limestone')
        """
        if len(predictions) != 10 or len(uncertainties) != 10:
            return {
                'matches_composition': False,
                'confidence': 0.0,
                'group_scores': {},
                'uncertainty_metrics': {
                    'mean_entropy': 0.0,
                    'mean_variance': 0.0,
                    'uncertain_predictions': 0
                }
            }

        # Count uncertain predictions
        uncertain_count = sum(
            1 for pred, unc in zip(predictions, uncertainties)
            if (unc['entropy'] > self.entropy_threshold or 
                unc['variance'] > self.variance_threshold or 
                pred == "unknown")
        )

        # Skip composition analysis if too many uncertain predictions
        if uncertain_count > 3:  # More than 30% uncertain
            return {
                'matches_composition': False,
                'confidence': 0.0,
                'group_scores': {},
                'uncertainty_metrics': {
                    'mean_entropy': np.mean([u['entropy'] for u in uncertainties]),
                    'mean_variance': np.mean([u['variance'] for u in uncertainties]),
                    'uncertain_predictions': uncertain_count
                }
            }

        group_scores = {}
        confidence_weights = {}
        predictions_lower = [p.lower() for p in predictions]
        constraints = self.hierarchy.get_composition_constraints(rock_type)

        # Calculate certainty-weighted scores
        for mineral, uncertainty in zip(predictions_lower, uncertainties):
            if mineral == "unknown":
                continue
                
            mineral_details = self.hierarchy.get_mineral_details(mineral, rock_type)
            if mineral_details:
                group = mineral_details['parent_group']
                base_weight = mineral_details.get('parent_weight', 0.0)
                
                # Calculate confidence weight based on uncertainty metrics
                confidence_weight = self._calculate_confidence_weight(uncertainty)
                
                # Accumulate weighted scores
                group_scores[group] = group_scores.get(group, 0) + (base_weight * confidence_weight)
                confidence_weights[group] = confidence_weights.get(group, 0) + confidence_weight

        # Normalize scores by confidence weights
        for group in group_scores:
            if confidence_weights[group] > 0:
                group_scores[group] /= confidence_weights[group]

        # Check if scores are within constraints with uncertainty margins
        matches_constraints = self._check_constraints_with_uncertainty(
            group_scores, 
            constraints,
            np.mean([u['variance'] for u in uncertainties])
        )

        # Calculate overall confidence considering uncertainties
        base_confidence = sum(group_scores.values()) / len(constraints)
        uncertainty_penalty = (uncertain_count / 10) * 0.5  # Up to 50% confidence reduction
        adjusted_confidence = base_confidence * (1 - uncertainty_penalty)

        return {
            'matches_composition': matches_constraints,
            'confidence': adjusted_confidence,
            'group_scores': group_scores,
            'uncertainty_metrics': {
                'mean_entropy': np.mean([u['entropy'] for u in uncertainties]),
                'mean_variance': np.mean([u['variance'] for u in uncertainties]),
                'uncertain_predictions': uncertain_count
            }
        }

    def _calculate_confidence_weight(self, uncertainty: Dict[str, float]) -> float:
        """Calculate confidence weight based on uncertainty metrics"""
        entropy_factor = max(0, 1 - (uncertainty['entropy'] / self.entropy_threshold))
        variance_factor = max(0, 1 - (uncertainty['variance'] / self.variance_threshold))
        return min(entropy_factor, variance_factor)

    def _check_constraints_with_uncertainty(self, 
                                         scores: Dict[str, float], 
                                         constraints: Dict[str, Tuple[float, float]],
                                         mean_variance: float) -> bool:
        """Check if scores meet constraints while considering uncertainty margins"""
        # Add uncertainty margin based on mean variance
        margin = mean_variance * 2  # Adjust multiplier based on validation
        
        for group, (min_val, max_val) in constraints.items():
            if group not in scores:
                return False
            score = scores[group]
            # Expand acceptable range by margin
            if not (min_val - margin <= score <= max_val + margin):
                return False
        return True

    def process_spectrum(self, spectrum: torch.Tensor, true_mineral: str = None) -> Dict:
        """Override process_spectrum to include uncertainty in rock analysis"""
        predicted_mineral, probabilities, entropy, variance = self.predict_mineral(spectrum)
        
        self.prediction_history.append(predicted_mineral)
        self.uncertainty_history.append({'entropy': entropy, 'variance': variance})
        if true_mineral is not None:
            self.ground_truth_history.append(true_mineral)
        
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
            self.uncertainty_history.pop(0)
            if self.ground_truth_history:
                self.ground_truth_history.pop(0)
        
        if len(self.prediction_history) == 10:
            # Evaluate compositions for each rock type
            granite_analysis = self.evaluate_rock_composition(
                self.prediction_history, 
                self.uncertainty_history,
                'granite'
            )
            sandstone_analysis = self.evaluate_rock_composition(
                self.prediction_history,
                self.uncertainty_history,
                'sandstone'
            )
            limestone_analysis = self.evaluate_rock_composition(
                self.prediction_history,
                self.uncertainty_history,
                'limestone'
            )
            
            # Determine most likely rock type
            rock_analyses = {
                RockType.GRANITE: granite_analysis,
                RockType.SANDSTONE: sandstone_analysis,
                RockType.LIMESTONE: limestone_analysis
            }
            
            best_rock_type = max(
                rock_analyses.items(),
                key=lambda x: x[1]['confidence'] if x[1]['matches_composition'] else -1
            )
            
            classification = (
                best_rock_type[0] 
                if best_rock_type[1]['matches_composition'] 
                else RockType.OTHER
            )
            
            rock_analysis = {
                'classification': classification.value,
                'confidence_scores': {
                    'granite': granite_analysis['confidence'],
                    'sandstone': sandstone_analysis['confidence'],
                    'limestone': limestone_analysis['confidence']
                },
                'uncertainty_metrics': self.uncertainty_history[-1],
                'composition_analyses': {
                    'granite': granite_analysis,
                    'sandstone': sandstone_analysis,
                    'limestone': limestone_analysis
                }
            }
        else:
            rock_analysis = {
                'classification': RockType.OTHER.value,
                'status': 'Insufficient measurements',
                'current_count': len(self.prediction_history)
            }
        
        return {
            'mineral_prediction': predicted_mineral,
            'mineral_probabilities': probabilities,
            'uncertainty': {'entropy': entropy, 'variance': variance},
            'rock_analysis': rock_analysis
        }