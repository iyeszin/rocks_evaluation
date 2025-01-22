import torch
import numpy as np
from typing import Dict, List, Tuple

from ..utils.enums import RockType
from ..utils.data_classes import MineralGroups
from ..utils.visualization import plot_mineral_analysis

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