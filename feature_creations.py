import json
import sys
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
from typing import List, Tuple, Dict, Any




gen1_defense_chart = {
    'normal': {'Weaknesses': ['fighting'], 'Resistances': [], 'Immunities': ['ghost']},
    'fire': {'Weaknesses': ['water', 'ground', 'rock'], 'Resistances': ['fire', 'grass', 'bug'], 'Immunities': []},
    'water': {'Weaknesses': ['electric', 'grass'], 'Resistances': ['fire', 'water', 'ice'], 'Immunities': []},
    'electric': {'Weaknesses': ['ground'], 'Resistances': ['electric', 'flying'], 'Immunities': []},
    'grass': {'Weaknesses': ['fire', 'ice', 'poison', 'flying', 'bug'], 'Resistances': ['water', 'electric', 'grass', 'ground'], 'Immunities': []},
    'ice': {'Weaknesses': ['fire', 'fighting', 'rock'], 'Resistances': ['ice'], 'Immunities': []},
    'fighting': {'Weaknesses': ['flying', 'psychic'], 'Resistances': ['bug', 'rock'], 'Immunities': []},
    'poison': {'Weaknesses': ['ground', 'psychic', 'bug'], 'Resistances': ['fighting', 'poison', 'grass'], 'Immunities': []},
    'ground': {'Weaknesses': ['water', 'grass', 'ice'], 'Resistances': ['poison', 'rock'], 'Immunities': ['electric']},
    'flying': {'Weaknesses': ['electric', 'ice', 'rock'], 'Resistances': ['grass', 'fighting', 'bug'], 'Immunities': ['ground']},
    'psychic': {'Weaknesses': ['bug'], 'Resistances': ['fighting', 'psychic'], 'Immunities': ['ghost']},
    'bug': {'Weaknesses': ['fire', 'poison', 'flying', 'rock'], 'Resistances': ['grass', 'fighting', 'ground'], 'Immunities': []},
    'rock': {'Weaknesses': ['water', 'grass', 'fighting', 'ground'], 'Resistances': ['normal', 'fire', 'poison', 'flying'], 'Immunities': []},
    'ghost': {'Weaknesses': ['ghost'], 'Resistances': ['poison', 'bug'], 'Immunities': ['normal', 'fighting']},
    'dragon': {'Weaknesses': ['ice', 'dragon'], 'Resistances': ['fire', 'water', 'electric', 'grass'], 'Immunities': []},
    'notype': {'Weaknesses': [], 'Resistances': [], 'Immunities': []}
}



def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON lines file with error handling"""
    data = []
    print(f"Loading data from '{file_path}'...")
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON at line {line_num}: {e}", file=sys.stderr)
        print(f"Successfully loaded {len(data)} battles.")
    except FileNotFoundError:
        print(f"ERROR: Could not find file at '{file_path}'.", file=sys.stderr)
        
    return data

#Optimized vectorized operations for dynamic features
def calculate_summary_stats_vectorized(data_array: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Vectorized summary statistics calculation"""
    if len(data_array) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    return (
        float(data_array[-1]),
        float(np.mean(data_array)),
        float(np.max(data_array)),
        float(np.min(data_array)),
        float(np.std(data_array)) if data_array.size > 1 else 0.0
    )

def calculate_momentum_vectorized(data_array: np.ndarray, window: int = 10) -> float:
    """
    Vectorized momentum calculation.
    It's used for hp and fainted, we see how much they variates in the last 10 turns.
    """
    if len(data_array) < 2:
        return 0.0
    
    start_idx = max(0, len(data_array) - window)
    if start_idx >= len(data_array) - 1:
        return 0.0
    
    # Ensure denominator is not zero if window is too small but array has > 1 element
    denominator = (len(data_array) - 1 - start_idx)
    if denominator == 0:
        return 0.0
        
    return (data_array[-1] - data_array[start_idx]) / denominator


class FeatureExtractor:
    """Class containing all methods to abstract features"""
    
    def __init__(self):
        # The weight of each threat are based on how much that pokemon
        # was impactful in 1° generation battle
        self.threat_tiers = {
            'tauros': 3, 'snorlax': 2, 'chansey': 2, 'alakazam': 1,
            'starmie': 1, 'exeggutor': 1, 'rhydon': 1
        }
        # The types are chosen according to how much they are considered dangerous in 1° gen battle
        self.key_attacking_types = ['ice', 'electric', 'ground', 'rock', 'psychic']
        # The weight of each status are based on how much each status
        # was impactful in 1° gen. battle
        self.status_weights = {
            'frz': 7.0, 'slp': 4.0, 'par': 3.0,
            'brn': 1.5, 'psn': 0.5, 'tox': 0.5
        }
        self.boost_weights = {'spe': 3, 'atk': 2, 'spa': 2, 'def': 1, 'spd': 1}

    def extract_static_features(self, p1_team: List[Dict[str, Any]], p2_lead: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract all static features at once.
        Static features are not based on the battle itself but on the nature of the teams.
        """
        features = {}
        
        # Team stats
        team_stats = self._compute_team_stats(p1_team)
        features.update(team_stats)
        
        # Type matchups
        type_scores = self._compute_type_scores(p1_team)
        features.update(type_scores)
        
        # Threat assessment
        features['p1_threat_score'] = self._compute_threat_score(p1_team, p2_lead, timeline)
        
        # Speed advantage
        features['starting_speed_adv'] = self._compute_speed_advantage(p1_team, p2_lead, timeline)
        
        return features
    
    
    def _compute_team_stats(self, p1_team: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Vectorized team stats computation"""
        if not p1_team:
            return {f'p1_avg_{k}': 0.0 for k in ['hp', 'atk', 'def', 'spe', 'special', 'crit_rate']}
        
        # Convert to numpy for vectorized operations
        stats_matrix = np.array([
            [p.get('base_hp', 0), p.get('base_atk', 0), p.get('base_def', 0),
             p.get('base_spe', 0), p.get('base_spa', 0)]
            for p in p1_team
        ])
        
        avg_stats = np.mean(stats_matrix, axis=0)
        
        # Crit rate calculation (Base Speed / 512 in Gen 1)
        speeds = stats_matrix[:, 3]
        avg_crit_rate = np.mean(speeds / 512.0)
        high_crit_threats = np.sum(speeds >= 100) # Arbitrary threshold for high-speed threats
        
        return {
            'p1_avg_hp': avg_stats[0],
            'p1_avg_atk': avg_stats[1],
            'p1_avg_def': avg_stats[2],
            'p1_avg_spe': avg_stats[3],
            'p1_avg_special': avg_stats[4],
            'p1_avg_crit_rate': avg_crit_rate,
            'p1_high_crit_threats': float(high_crit_threats)
        }


    def _compute_type_scores(self, p1_team: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimized type score computation"""
        scores = {f'p1_score_vs_{atk_type}': 0 for atk_type in self.key_attacking_types}
        
        for pokemon in p1_team:
            types = pokemon.get('types', ['notype', 'notype'])
            # Ensure types list has at least two elements
            if len(types) == 1:
                types.append('notype')
            elif not types:
                types = ['notype', 'notype']

            for attacking_type in self.key_attacking_types:
                eff1 = self._get_effectiveness(attacking_type, types[0])
                eff2 = self._get_effectiveness(attacking_type, types[1] if types[1] else 'notype')
                
                # Combined effectiveness calculation
                combined_eff = eff1 * eff2
                
                if combined_eff == 0.0:
                    scores[f'p1_score_vs_{attacking_type}'] += 2 # Immunity
                elif combined_eff < 1.0:
                    scores[f'p1_score_vs_{attacking_type}'] += 1 # Resistance
                elif combined_eff > 1.0:
                    scores[f'p1_score_vs_{attacking_type}'] -= 1 # Weakness
        
        return scores


    def _get_effectiveness(self, attacking_type: str, defending_type: str) -> float:
        """Quick type effectiveness lookup"""
        defending_type = defending_type.lower() if defending_type else 'notype'
        attacking_type = attacking_type.lower() if attacking_type else 'notype'
        
        defense_info = gen1_defense_chart.get(defending_type, {})
        
        if attacking_type in defense_info.get('Immunities', []):
            return 0.0
        elif attacking_type in defense_info.get('Weaknesses', []):
            return 2.0
        elif attacking_type in defense_info.get('Resistances', []):
            return 0.5
        return 1.0


    def _compute_threat_score(self, p1_team: List[Dict[str, Any]], p2_lead: Dict[str, Any], timeline: List[Dict[str, Any]]) -> int:
        """Compute threat differential"""
        p1_names = {p.get('name', '').lower() for p in p1_team}
        p2_names = {p2_lead.get('name', '').lower()}
        
        # Get unique p2 pokemon from timeline
        for turn in timeline:
            p2_name = turn.get('p2_pokemon_state', {}).get('name', '').lower()
            if p2_name:
                p2_names.add(p2_name)
        
        p1_score = sum(self.threat_tiers.get(name, 0) for name in p1_names)
        p2_score = sum(self.threat_tiers.get(name, 0) for name in p2_names)
        
        return p1_score - p2_score
    
    def _compute_speed_advantage(self, p1_team: List[Dict[str, Any]], p2_lead: Dict[str, Any], timeline: List[Dict[str, Any]]) -> int:
        """Compute starting speed advantage"""
        if not timeline:
            return 0
        
        p1_starting = timeline[0].get('p1_pokemon_state', {}).get('name')
        p2_lead_spe = p2_lead.get('base_spe', 0)
        
        for pokemon in p1_team:
            if pokemon.get('name') == p1_starting:
                return pokemon.get('base_spe', 0) - p2_lead_spe
        
        return 0
    
    def extract_dynamic_features(self, timeline: List[Dict[str, Any]], p1_team: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract all dynamic features efficiently.
        Dynamic features are features based on the battle itself.
        """
        features = {}
        
        # Prepare timeline arrays
        timeline_arrays = self._build_timeline_arrays(timeline, p1_team)
        
        # Calculate summary stats for each array
        for name, array in timeline_arrays.items():
            stats = calculate_summary_stats_vectorized(array)
            features[f'{name}_last'] = stats[0]
            features[f'{name}_avg'] = stats[1]
            features[f'{name}_max'] = stats[2]
            features[f'{name}_min'] = stats[3]
            features[f'{name}_std'] = stats[4]
            
        # Calculate momentum for key features
        features['fainted_momentum'] = calculate_momentum_vectorized(timeline_arrays['fainted'])
        features['hpdiff_momentum'] = calculate_momentum_vectorized(timeline_arrays['hpdiff'])
        
        return features


    def _build_timeline_arrays(self, timeline: List[Dict[str, Any]], p1_team: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Build all timeline arrays in a single pass"""
        max_turns = min(len(timeline), Config.MAX_TURNS)
        
        # Initialize arrays
        arrays = {
            'fainted': np.zeros(max_turns),
            'hpdiff': np.zeros(max_turns),
            'lowhp': np.zeros(max_turns),
            'status': np.zeros(max_turns),
            'boost': np.zeros(max_turns),
            'recovery_diff': np.zeros(max_turns),
            'chip_diff': np.zeros(max_turns),
            'pressure_diff': np.zeros(max_turns)
        }
        # State tracking
        p1_kos, p2_kos = 0, 0
        p1_recovery, p2_recovery = 0, 0
        p1_chip, p2_chip = 0, 0
        p1_pressure, p2_pressure = 0, 0
        
        p1_team_names = {p.get('name') for p in p1_team}
        p1_hp_status = {p.get('name'): 1.0 for p in p1_team}
        p2_hp_status = {}
        p1_team_status = {p.get('name'): 'nostatus' for p in p1_team}
        p2_team_status = {}
        
        prev_p1_status, prev_p2_status = 'nostatus', 'nostatus'
        
        recovery_moves = {'recover', 'soft-boiled', 'rest'}
        chip_moves = {'toxic', 'leech seed', 'wrap', 'bind', 'fire spin', 'clamp'}
        pressure_moves = {'blizzard', 'psychic', 'thunderbolt', 'surf', 'earthquake', 'body slam', 'hyper beam'}
    
        for i, turn in enumerate(timeline[:max_turns]):
            p1_state = turn.get('p1_pokemon_state', {})
            p2_state = turn.get('p2_pokemon_state', {})
            
            # KO tracking
            current_p1_status = p1_state.get('status', 'nostatus')
            current_p2_status = p2_state.get('status', 'nostatus')
            if current_p1_status == 'fnt' and prev_p1_status != 'fnt':
                p1_kos += 1
            if current_p2_status == 'fnt' and prev_p2_status != 'fnt':
                p2_kos += 1
            arrays['fainted'][i] = p2_kos - p1_kos
                
            # HP tracking
            p1_name = p1_state.get('name')
            p2_name = p2_state.get('name')
            if p1_name:
                p1_hp_status[p1_name] = p1_state.get('hp_pct', p1_hp_status.get(p1_name, 0.0))
            if p2_name:
                if p2_name not in p2_hp_status:
                    p2_hp_status[p2_name] = 1.0
                p2_hp_status[p2_name] = p2_state.get('hp_pct', p2_hp_status.get(p2_name, 0.0))
            
            p1_total_hp = sum(v for k, v in p1_hp_status.items() if k in p1_team_names)
            # Assume opponent has 6 Pokémon unless revealed otherwise
            p2_total_hp = sum(p2_hp_status.values()) + (6 - len(p2_hp_status))
            arrays['hpdiff'][i] = (p1_total_hp / 6.0) - (p2_total_hp / 6.0)
            
            # Low HP tracking
            p1_low = sum(1 for k, hp in p1_hp_status.items() if k in p1_team_names and 0 < hp < 0.5)
            p2_low = sum(1 for hp in p2_hp_status.values() if 0 < hp < 0.5)
            arrays['lowhp'][i] = p2_low - p1_low
            
            # Status tracking
            if p1_name:
                p1_team_status[p1_name] = current_p1_status
            if p2_name:
                p2_team_status[p2_name] = current_p2_status
            
            p1_stat_score = sum(
                self.status_weights.get(stat, 0) 
                for k, stat in p1_team_status.items() if k in p1_team_names
            )
            p2_stat_score = sum(
                self.status_weights.get(stat, 0) 
                for stat in p2_team_status.values()
            )
            arrays['status'][i] = p2_stat_score - p1_stat_score
            
            # Boost tracking
            p1_boosts = p1_state.get('boosts', {})
            p2_boosts = p2_state.get('boosts', {})
            p1_boost_score = sum(
                self.boost_weights.get(stat, 0) * level 
                for stat, level in p1_boosts.items()
            )
            p2_boost_score = sum(
                self.boost_weights.get(stat, 0) * level 
                for stat, level in p2_boosts.items()
            )
            arrays['boost'][i] = p1_boost_score - p2_boost_score
            
            # Move tracking
            p1_move = (turn.get('p1_move_details') or {}).get('name', '').lower()
            p2_move = (turn.get('p2_move_details') or {}).get('name', '').lower()
            
            if p1_move in recovery_moves:
                p1_recovery += 1
            if p2_move in recovery_moves:
                p2_recovery += 1
            arrays['recovery_diff'][i] = p1_recovery - p2_recovery
            
            if p1_move in chip_moves:
                p1_chip += 1
            if p2_move in chip_moves:
                p2_chip += 1
            arrays['chip_diff'][i] = p1_chip - p2_chip
            
            if p1_move in pressure_moves:
                p1_pressure += 1
            if p2_move in pressure_moves:
                p2_pressure += 1
            arrays['pressure_diff'][i] = p1_pressure - p2_pressure
            
            prev_p1_status = current_p1_status
            prev_p2_status = current_p2_status
        
        # Pad arrays if needed
        if max_turns < Config.MAX_TURNS:
            for key in arrays:
                last_val = arrays[key][-1] if max_turns > 0 else 0
                padded = np.full(Config.MAX_TURNS, last_val)
                padded[:max_turns] = arrays[key]
                arrays[key] = padded
        
        return arrays


def create_features(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Main feature creation function"""
    
    extractor = FeatureExtractor()
    feature_list = []
    
    for battle in tqdm(data, desc="Extracting features"):
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})
        timeline = battle.get('battle_timeline', [])
        
        # Skip invalid battles
        if not p1_team or not p2_lead or not timeline or len(p1_team) != 6:
            continue
        
        try:
            features = {}
            
            # Extract static features
            static_features = extractor.extract_static_features(p1_team, p2_lead, timeline[:Config.MAX_TURNS])
            features.update(static_features)
            
            # Extract dynamic features
            dynamic_features = extractor.extract_dynamic_features(timeline[:Config.MAX_TURNS], p1_team)
            features.update(dynamic_features)
            
            # Add metadata
            features[Config.ID_COLUMN_NAME] = battle.get(Config.ID_COLUMN_NAME)
            if Config.TARGET_COLUMN_NAME in battle:
                features[Config.TARGET_COLUMN_NAME] = int(battle[Config.TARGET_COLUMN_NAME])
            
            feature_list.append(features)
            
        except Exception as e:
            print(f"Error processing battle {battle.get(Config.ID_COLUMN_NAME)}: {e}")
            continue
    
    df = pd.DataFrame(feature_list)
    
    return df
