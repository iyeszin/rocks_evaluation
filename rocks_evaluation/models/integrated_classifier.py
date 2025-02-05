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
    def __init__(self, model, label_encoder, device, window_size: int = 10, 
                 uncertainty_threshold: float = 0.2):
        """
        Initialize the uncertainty-aware hierarchical rock classifier
        
        Args:
            model: The trained neural network model
            label_encoder: Encoder for mineral labels
            device: Device to run model on ('cuda' or 'cpu')
            window_size: Number of measurements to use for rock type classification
            uncertainty_threshold: Threshold for considering predictions uncertain
        """
        super().__init__(model, label_encoder, device, window_size)
        self.hierarchy = MineralHierarchy()
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_history = []  # Track uncertainties over time
        
    def predict_mineral_with_uncertainty(self, spectrum: torch.Tensor) -> Tuple[str, Dict[str, float], float]:
        """
        Predict mineral and estimate uncertainty using entropy-based approach
        
        Args:
            spectrum: Input spectrum tensor
            
        Returns:
            Tuple of (predicted_mineral, probabilities_dict, uncertainty)
        """
        # Move input to device and add batch dimension if needed
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
        spectrum = spectrum.to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(spectrum)
            probabilities = torch.softmax(logits, dim=1)
        
        # Convert to numpy for calculations
        probs_np = probabilities.cpu().numpy()[0]
        
        # Calculate entropy-based uncertainty
        epsilon = 1e-10  # Small constant to avoid log(0)
        entropy = -np.sum(probs_np * np.log(probs_np + epsilon))
        max_entropy = -np.log(1.0/len(self.label_encoder.classes_))  # Maximum possible entropy
        uncertainty = entropy / max_entropy  # Normalize to [0,1]
        
        # Get predicted class
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_mineral = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        # Create probabilities dictionary
        probabilities_dict = {
            mineral: float(prob)
            for mineral, prob in zip(self.label_encoder.classes_, probs_np)
        }
        
        return predicted_mineral, probabilities_dict, uncertainty

    def get_mineral_counts(self, predictions: List[str], uncertainties: List[float]) -> Dict[str, float]:
        """
        Count occurrences of each mineral in predictions, weighted by certainty
        
        Args:
            predictions: List of predicted mineral names
            uncertainties: List of uncertainty values for each prediction
        
        Returns:
            Dictionary of weighted counts for each mineral group
        """
        predictions_lower = [p.lower() for p in predictions]
        certainties = [1 - u for u in uncertainties]  # Convert uncertainties to certainty weights
        
        # Initialize counters with weighted counts
        counts = {
            'quartz': 0.0,
            'feldspars': 0.0,
            'micas': 0.0,
            'calcite': 0.0,
            'pyrite': 0.0,
            'rutile': 0.0,
            'tourmaline': 0.0
        }
        
        for pred, cert in zip(predictions_lower, certainties):
            # Add weighted counts based on prediction certainty
            if pred == 'quartz':
                counts['quartz'] += cert
            elif pred in ['orthoclase', 'albite', 'anorthite']:
                counts['feldspars'] += cert
            elif pred in ['annite', 'phlogopite', 'muscovite']:
                counts['micas'] += cert
            elif pred == 'calcite':
                counts['calcite'] += cert
            elif pred == 'pyrite':
                counts['pyrite'] += cert
            elif pred == 'rutile':
                counts['rutile'] += cert
            elif pred == 'tourmaline':
                counts['tourmaline'] += cert
                
        return counts

    def evaluate_rock_type_weights(self, predictions: List[str], 
                                 uncertainties: List[float]) -> Dict[str, float]:
        """
        Evaluate likelihood weights for each rock type based on mineral proportions,
        taking into account prediction uncertainties
        """
        if len(predictions) != self.window_size:
            return {'granite': 0.0, 'sandstone': 0.0, 'limestone': 0.0}
        
        counts = self.get_mineral_counts(predictions, uncertainties)
        total_certainty = sum(1 - u for u in uncertainties)  # Sum of certainty weights
        
        # Avoid division by zero
        if total_certainty == 0:
            return {'granite': 0.0, 'sandstone': 0.0, 'limestone': 0.0}
        
        # Calculate certainty-weighted ratios
        quartz_ratio = counts['quartz'] / total_certainty
        feldspar_ratio = counts['feldspars'] / total_certainty
        mica_ratio = counts['micas'] / total_certainty
        calcite_ratio = counts['calcite'] / total_certainty
        
        weights = {
            'granite': 0.0,
            'sandstone': 0.0,
            'limestone': 0.0
        }
        
        # Granite scoring with uncertainty consideration
        granite_score = 0.0
        if 0.2 <= quartz_ratio <= 0.4:
            granite_score += 0.4
        if 0.35 <= feldspar_ratio <= 0.8:
            granite_score += 0.4
        if 0.0 <= mica_ratio <= 0.35:
            granite_score += 0.2
        weights['granite'] = granite_score
        
        # Sandstone scoring
        sandstone_score = 0.0
        if quartz_ratio >= 0.65:
            sandstone_score += 0.5
        if 0.05 <= feldspar_ratio <= 0.25:
            sandstone_score += 0.3
        if calcite_ratio <= 0.1:
            sandstone_score += 0.2
        weights['sandstone'] = sandstone_score
        
        # Limestone scoring
        limestone_score = 0.0
        if calcite_ratio >= 0.85:
            limestone_score = 1.0
        elif calcite_ratio >= 0.45:
            limestone_score = 0.8
        weights['limestone'] = limestone_score
        
        # Apply overall uncertainty penalty
        mean_uncertainty = sum(uncertainties) / len(uncertainties)
        uncertainty_factor = 1 - mean_uncertainty
        weights = {k: v * uncertainty_factor for k, v in weights.items()}
        
        # Normalize weights if any are non-zero
        max_weight = max(weights.values())
        if max_weight > 0:
            weights = {k: v/max_weight for k, v in weights.items()}
            
        return weights

    def check_mineral_assemblage_rules(self, predictions: List[str], 
                                     uncertainties: List[float]) -> Dict:
        """
        Check mineral assemblages using weight-based approach with uncertainty consideration
        """
        weights = self.evaluate_rock_type_weights(predictions, uncertainties)
        
        # Adjust thresholds based on uncertainty
        mean_uncertainty = sum(uncertainties) / len(uncertainties)
        adjusted_confidence_threshold = self.uncertainty_threshold + (
            0.7 * (1 - mean_uncertainty)  # Base threshold of 0.7 adjusted by certainty
        )
        adjusted_dominance_threshold = 0.3 * (1 - mean_uncertainty)
        
        # Get the two highest weights
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        highest_weight = sorted_weights[0][1]
        second_highest = sorted_weights[1][1] if len(sorted_weights) > 1 else 0
        
        # Check if we're confident enough given uncertainty
        is_confident = (highest_weight >= adjusted_confidence_threshold and 
                       (highest_weight - second_highest) >= adjusted_dominance_threshold)
        
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
            'counts': self.get_mineral_counts(predictions, uncertainties),
            'mean_uncertainty': mean_uncertainty,
            'adjusted_confidence_threshold': adjusted_confidence_threshold
        }

    def process_spectrum(self, spectrum: torch.Tensor, true_mineral: str = None) -> Dict:
        """Process spectrum and determine rock type with uncertainty consideration"""
        # Get mineral prediction and uncertainty
        predicted_mineral, probabilities, uncertainty = self.predict_mineral_with_uncertainty(spectrum)
        
        self.prediction_history.append(predicted_mineral)
        self.uncertainty_history.append(uncertainty)
        if true_mineral is not None:
            self.ground_truth_history.append(true_mineral)
        
        # Maintain window size
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
            self.uncertainty_history.pop(0)
            if self.ground_truth_history:
                self.ground_truth_history.pop(0)
        
        # Perform rock type analysis if we have enough measurements
        if len(self.prediction_history) == self.window_size:
            mineral_analysis = self.check_mineral_assemblage_rules(
                self.prediction_history, 
                self.uncertainty_history
            )
            
            # Determine classification based on weights and uncertainty
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
                'is_confident': mineral_analysis['satisfied'],
                'mean_uncertainty': mineral_analysis['mean_uncertainty'],
                'confidence_threshold': mineral_analysis['adjusted_confidence_threshold']
            }
        else:
            rock_analysis = {
                'classification': RockType.OTHER.value,
                'status': 'Insufficient measurements',
                'current_count': len(self.prediction_history)
            }
        
        # Store in analysis_history
        self.analysis_history.append({
            'mineral_prediction': predicted_mineral,
            'true_mineral': true_mineral,
            'uncertainty': uncertainty,
            'rock_analysis': rock_analysis,
            'measurement_number': len(self.prediction_history)
        })
        
        return {
            'mineral_prediction': predicted_mineral,
            'mineral_probabilities': probabilities,
            'uncertainty': uncertainty,
            'rock_analysis': rock_analysis
        }