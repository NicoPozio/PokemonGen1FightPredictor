import json
import sys
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config

#Gen1 Types relationship Chart
gen1_defense_chart = {
    'normal': { 'Weaknesses': ['fighting'], 'Resistances': [], 'Immunities': ['ghost'] },
    'fire': { 'Weaknesses': ['water', 'ground', 'rock'], 'Resistances': ['fire', 'grass', 'bug'], 'Immunities': [] },
    'water': { 'Weaknesses': ['electric', 'grass'], 'Resistances': ['fire', 'water', 'ice'], 'Immunities': [] },
    'electric': { 'Weaknesses': ['ground'], 'Resistances': ['electric', 'flying'], 'Immunities': [] },
    'grass': { 'Weaknesses': ['fire', 'ice', 'poison', 'flying', 'bug'], 'Resistances': ['water', 'electric', 'grass', 'ground'], 'Immunities': [] },
    'ice': { 'Weaknesses': ['fire', 'fighting', 'rock'], 'Resistances': ['ice'], 'Immunities': [] },
    'fighting': { 'Weaknesses': ['flying', 'psychic'], 'Resistances': ['bug', 'rock'], 'Immunities': [] },
    'poison': { 'Weaknesses': ['ground', 'psychic', 'bug'], 'Resistances': ['fighting', 'poison', 'grass'], 'Immunities': [] },
    'ground': { 'Weaknesses': ['water', 'grass', 'ice'], 'Resistances': ['poison', 'rock'], 'Immunities': ['electric'] },
    'flying': { 'Weaknesses': ['electric', 'ice', 'rock'], 'Resistances': ['grass', 'fighting', 'bug'], 'Immunities': ['ground'] },
    'psychic': { 'Weaknesses': ['bug'], 'Resistances': ['fighting', 'psychic'], 'Immunities': ['ghost'] },
    'bug': { 'Weaknesses': ['fire', 'poison', 'flying', 'rock'], 'Resistances': ['grass', 'fighting', 'ground'], 'Immunities': [] },
    'rock': { 'Weaknesses': ['water', 'grass', 'fighting', 'ground'], 'Resistances': ['normal', 'fire', 'poison', 'flying'], 'Immunities': [] },
    'ghost': { 'Weaknesses': ['ghost'], 'Resistances': ['poison', 'bug'], 'Immunities': ['normal', 'fighting'] },
    'dragon': { 'Weaknesses': ['ice', 'dragon'], 'Resistances': ['fire', 'water', 'electric', 'grass'], 'Immunities': [] },
    'notype': { 'Weaknesses': [], 'Resistances': [], 'Immunities': [] }
}



def load_data(file_path: str) -> list[dict]:
    """Load a json file"""
    data = []
    print(f"Caricamento dati da '{file_path}'...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}", file=sys.stderr)
        print(f"Successfully loaded {len(data)} battles.")
    except FileNotFoundError:
        print(f"ERROR: Could not find file at '{file_path}'.", file=sys.stderr)
    return data


def calculate_summary_stats(data_array: list) -> tuple:
    """
    Given an array with info on the battle, 
    it computes summary stats useful to better understand the evolution of the battle
    """
    np_array = np.array(data_array, dtype=float)
    if len(np_array) == 0: return (0.0, 0.0, 0.0, 0.0, 0.0)
    last_val = np_array[-1]
    mean_val = np.mean(np_array)
    max_val = np.max(np_array)
    min_val = np.min(np_array)
    std_val = np.std(np_array) if np_array.size > 1 else 0.0
    return (float(last_val), float(mean_val), float(max_val), float(min_val), float(std_val))

def pad_array(data_list: list, pad_length: int) -> list:
    """
    If for some case we examinate a timeline with less than 30 turns
    this function extends the array with info padding it with the last value
    """
    actual_length = len(data_list)
    if actual_length == 0: return [0.0] * pad_length
    if actual_length >= pad_length: return data_list[:pad_length]
    last_value = data_list[-1]
    padding = [float(last_value)] * (pad_length - actual_length)
    return data_list + padding
    
def calculate_momentum(data_array: list, window: int = 10) -> float:
    """
    This function measures how much a certain feature changed in the last 10
    turns of the timeline
    """
    np_array = np.array(data_array)
    actual_length = len(np_array)
    if actual_length < 2 or window < 1: return 0.0
    start_index = max(0, actual_length - window)
    end_index = actual_length - 1
    if start_index >= end_index:
        if actual_length > 1:
            start_index = 0
            duration = end_index - start_index
            value_change = np_array[end_index] - np_array[start_index]
            return value_change / float(duration) if duration > 0 else 0.0
        else: return 0.0
    duration = end_index - start_index
    value_change = np_array[end_index] - np_array[start_index]
    return value_change / float(duration) if duration > 0 else 0.0



def get_avg_stats(p1_team: list[dict]) -> dict[str, float]:
    """This function computes the average stats for p1's team"""
    stats = {'hp': 0.0, 'atk': 0.0, 'def': 0.0, 'spe': 0.0, 'spa': 0.0}
    num_pokemon = len(p1_team)
    if num_pokemon == 0: return {f'p1_avg_{k}': 0.0 for k in stats}
    for p in p1_team:
        stats['hp'] += p.get('base_hp', 0)
        stats['atk'] += p.get('base_atk', 0)
        stats['def'] += p.get('base_def', 0)
        stats['spe'] += p.get('base_spe', 0)
        stats['spa'] += p.get('base_spa', 0)
    avg_stats_features = {
        'p1_avg_hp': stats['hp'] / num_pokemon,
        'p1_avg_atk': stats['atk'] / num_pokemon,
        'p1_avg_def': stats['def'] / num_pokemon,
        'p1_avg_spe': stats['spe'] / num_pokemon,
        'p1_avg_special': stats['spa'] / num_pokemon
    }
    return avg_stats_features

def get_key_threat_score(p1_team: list[dict], p2_lead: dict, timeline: list[dict]) -> int:
    """
    This function exploit the knowledge about stronget pokemon in Gen1 competitive battle
    and computes the number of key_threat for p1 and p2, returning the difference
    """
    threat_tiers = {
        'tauros': 3, 'snorlax': 2, 'chansey': 2, 'alakazam': 1,
        'starmie': 1, 'exeggutor': 1, 'zapdos': 1, 'rhydon': 1, 'jolteon': 1
    }
    p1_names = {p.get('name', '').lower() for p in p1_team}
    p2_names = {p2_lead.get('name', '').lower()}
    for turn in timeline:
        p2_name = turn['p2_pokemon_state'].get('name', '').lower()
        if p2_name: p2_names.add(p2_name)
    p1_score = sum(threat_tiers.get(name, 0) for name in p1_names)
    p2_score = sum(threat_tiers.get(name, 0) for name in p2_names)
    return p1_score - p2_score

def get_effectiveness(attacking_type: str, defending_type: str) -> float:
    """This function return how much the defending type is weak to the attacking tipe"""
    defending_type = defending_type.lower()
    attacking_type = attacking_type.lower()
    
    if not defending_type or defending_type == 'notype': return 1.0
        
    defense_info = gen1_defense_chart.get(defending_type, {})
    
    if not defense_info: return 1.0
    if attacking_type in defense_info.get('Immunities', []): return 0.0
    elif attacking_type in defense_info.get('Weaknesses', []): return 2.0
    elif attacking_type in defense_info.get('Resistances', []): return 0.5
        
    else: return 1.0

def get_team_type_scores(p1_team: list[dict]) -> dict[str, int]:
    """
    This function exploit the knowledge about strongest and
    most used types in Gen1 competitive battle
    to measure how much p1_team is weak to these type
    """
    key_attacking_types = ['ice', 'electric', 'ground', 'rock', 'psychic']
    team_scores = {f'p1_score_vs_{atk_type}': 0 for atk_type in key_attacking_types}
    for pokemon in p1_team:
        type1 = pokemon['types'][0]
        type2 = pokemon['types'][1]
        for attacking_type in key_attacking_types:
            eff1 = get_effectiveness(attacking_type, type1)
            eff2 = get_effectiveness(attacking_type, type2)
            combined_eff = 0.0 if (eff1 == 0.0 or eff2 == 0.0) else eff1 * eff2
            score = 0
            if combined_eff == 0.0: score = 2
            elif combined_eff < 1.0: score = 1
            elif combined_eff > 1.0: score = -1
            team_scores[f'p1_score_vs_{attacking_type}'] += score
    return team_scores

def get_starting_superiority(p1_team: list[dict], p2_lead: dict, battle_timeline: list[dict]) -> int:
    """
    This function computes the difference of speed between starting pokemon
    """
    if not battle_timeline: return 0
    p1_state_turn0 = battle_timeline[0].get('p1_pokemon_state', {})
    p1_starting_pokemon_name = p1_state_turn0.get('name')
    p2_lead_spe = p2_lead.get('base_spe', 0)
    p1_lead_spe = 0
    if p1_starting_pokemon_name:
        for pokemon in p1_team:
            if pokemon.get('name') == p1_starting_pokemon_name:
                p1_lead_spe = pokemon.get('base_spe', 0); break
    return p1_lead_spe - p2_lead_spe


def get_avg_crit_rate(p1_team: list[dict]) -> dict[str, float]:
    """
    This function computes average critic rate of p1_team (based on speed)
    and how many pokemon there are in p1_team that have a good critic rate (speed>=100)
    """
    if not p1_team:
        return {'p1_avg_crit_rate': 0.0, 'p1_high_crit_threats': 0}
    total_crit_rate = 0.0
    high_crit_threats = 0
    num_pokemon = len(p1_team)
    if num_pokemon == 0:
        return {'p1_avg_crit_rate': 0.0, 'p1_high_crit_threats': 0}
        
    for p in p1_team:
        base_spe = p.get('base_spe', 0)
        # Critic Rate Gen 1 = Base Speed / 512
        total_crit_rate += (base_spe / 512.0)
        if base_spe >= 100:
            high_crit_threats += 1
                
    return {
        'p1_avg_crit_rate': total_crit_rate / num_pokemon,
        'p1_high_crit_threats': float(high_crit_threats)
    }


def get_fainted_array(timeline: list[dict]) -> list[float]:
    """
    This function generates an array that computes
    the difference of KO between p1 and p2
    """
    fainted_diff_raw = []
    p1_kos, p2_kos = 0, 0
    prev_p1_status, prev_p2_status = 'nostatus', 'nostatus'
    for turn in timeline:
        current_p1_status = turn.get('p1_pokemon_state', {}).get('status', 'nostatus')
        current_p2_status = turn.get('p2_pokemon_state', {}).get('status', 'nostatus')
        if current_p1_status == 'fnt' and prev_p1_status != 'fnt': p1_kos += 1
        if current_p2_status == 'fnt' and prev_p2_status != 'fnt': p2_kos += 1
        fainted_diff_raw.append(float(p2_kos - p1_kos))
        prev_p1_status, prev_p2_status = current_p1_status, current_p2_status
    return pad_array(fainted_diff_raw, Config.MAX_TURNS)

def get_hp_arrays(timeline: list[dict], p1_team: list[dict]) -> tuple[list[float], list[float]]:
    """
    This function generates an array that computes
    the difference of HP_pct between p1 and p2
    """
    hp_diff_raw, low_hp_diff_raw = [], []
    p1_hp_status = {p.get('name'): 1.0 for p in p1_team if p.get('name')}
    p2_hp_status = {}
    p1_team_names = {p.get('name') for p in p1_team if p.get('name')}
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})
        p1_name, p2_name = p1_state.get('name'), p2_state.get('name')
        if p1_name: p1_hp_status[p1_name] = p1_state.get('hp_pct', p1_hp_status.get(p1_name, 0.0))
        if p2_name:
            if p2_name not in p2_hp_status: p2_hp_status[p2_name] = 1.0
            p2_hp_status[p2_name] = p2_state.get('hp_pct', p2_hp_status.get(p2_name, 0.0))
        
        p1_total_hp = sum(v for k,v in p1_hp_status.items() if k in p1_team_names)

        p2_total_hp = sum(p2_hp_status.values()) + (6 - len(p2_hp_status)) * 1.0
        
        hp_diff_raw.append((p1_total_hp / 6.0) - (p2_total_hp / 6.0))
        
        p1_low_hp = sum(1 for k, hp in p1_hp_status.items() if k in p1_team_names and 0 < hp < 0.5)
        p2_low_hp = sum(1 for hp in p2_hp_status.values() if 0 < hp < 0.5)
        low_hp_diff_raw.append(float(p2_low_hp - p1_low_hp))
    return pad_array(hp_diff_raw, Config.MAX_TURNS), pad_array(low_hp_diff_raw, Config.MAX_TURNS)

def get_status_array(timeline: list[dict], p1_team: list[dict]) -> list[float]:
    """
    This function generates an array that computes
    the difference of inflicted status between p1 and p2.
    Each status has a specific weight, set according to how much
    that state it's powerful in Gen1 competitive
    """
    status_diff_raw = []
    p1_team_status = {p.get('name'): 'nostatus' for p in p1_team if p.get('name')}
    p2_team_status = {}
    
    status_weights = {
        'frz': 6.0, 'slp': 4.0, 'par': 3.0,
        'brn': 1.0, 'psn': 0.5, 'tox': 0.5
    }
    p1_team_names = {p.get('name') for p in p1_team if p.get('name')}
    
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})
        p1_name, p2_name = p1_state.get('name'), p2_state.get('name')
        current_p1_status = p1_state.get('status', 'nostatus')
        current_p2_status = p2_state.get('status', 'nostatus')
        
        if p1_name: p1_team_status[p1_name] = current_p1_status
        if p2_name:
            if p2_name not in p2_team_status: p2_team_status[p2_name] = 'nostatus'
            p2_team_status[p2_name] = current_p2_status
            
        p1_stat_score = sum(status_weights.get(stat, 0) for k, stat in p1_team_status.items() if k in p1_team_names)

        
        p2_stat_score = sum(status_weights.get(stat, 0) for stat in p2_team_status.values())
        status_diff_raw.append(float(p2_stat_score - p1_stat_score))
    return pad_array(status_diff_raw, Config.MAX_TURNS)

def get_boost_array(timeline: list[dict]) -> list[float]:
    """    
    This function generates an array that computes
    the difference of boosts between p1 and p2
    """
    boost_diff_raw = []
    boost_weights = {'spe': 3, 'atk': 2, 'spa': 2, 'def': 1, 'spd': 1}
    for turn in timeline:
        p1_boosts = turn.get('p1_pokemon_state', {}).get('boosts', {})
        p2_boosts = turn.get('p2_pokemon_state', {}).get('boosts', {})
        p1_boost_score = sum(boost_weights.get(stat, 0) * level for stat, level in p1_boosts.items())
        p2_boost_score = sum(boost_weights.get(stat, 0) * level for stat, level in p2_boosts.items())
        boost_diff_raw.append(float(p1_boost_score - p2_boost_score))
    return pad_array(boost_diff_raw, Config.MAX_TURNS)

def get_switch_array(timeline: list[dict]) -> list[float]:
    """
    This function generates an array that computes
    the difference of switches between p1 and p2
    (Numerous switch can mean that the player is in difficulty)
    """
    switch_diff_raw = []
    p1_switch_count, p2_switch_count = 0, 0
    prev_p1_name, prev_p2_name = None, None
    prev_p1_status, prev_p2_status = 'nostatus', 'nostatus'
    if timeline:
        init_p1_state = timeline[0].get('p1_pokemon_state', {})
        init_p2_state = timeline[0].get('p2_pokemon_state', {})
        prev_p1_name = init_p1_state.get('name')
        prev_p2_name = init_p2_state.get('name')
    for i, turn in enumerate(timeline):
        p1_state = turn.get('p1_pokemon_state', {})
        p2_state = turn.get('p2_pokemon_state', {})
        current_p1_name = p1_state.get('name')
        current_p2_name = p2_state.get('name')
        current_p1_status = p1_state.get('status', 'nostatus')
        current_p2_status = p2_state.get('status', 'nostatus')
        if i > 0:
            if current_p1_name != prev_p1_name and prev_p1_status != 'fnt': p1_switch_count += 1
            if current_p2_name != prev_p2_name and prev_p2_status != 'fnt': p2_switch_count += 1
        switch_diff_raw.append(float(p1_switch_count - p2_switch_count))
        prev_p1_name, prev_p2_name = current_p1_name, current_p2_name
        prev_p1_status, prev_p2_status = current_p1_status, current_p2_status
    return pad_array(switch_diff_raw, Config.MAX_TURNS)

def get_p1_stab_array(timeline: list[dict], p1_team: list[dict]) -> list[float]:
    """
    This function generates an array that computes
    the number of STAB moves (move's type = pokemon's type) used by p1
    For p2 we cannot compute this value.
    """
    p1_stab_count_raw = []
    p1_stab_cumulative = 0
    p1_types = {p['name']: p['types'] for p in p1_team}
    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state', {})
        p1_move = turn.get('p1_move_details', {})
        current_p1_name = p1_state.get('name')
        p1_used_stab = False
        if p1_move and current_p1_name in p1_types:
            move_type = p1_move.get('type', '').lower()
            p1_type = p1_types[current_p1_name]
            if move_type and (move_type == p1_type[0] or move_type == p1_type[1]):
                p1_used_stab = True
        if p1_used_stab: p1_stab_cumulative += 1
        p1_stab_count_raw.append(float(p1_stab_cumulative))
    return pad_array(p1_stab_count_raw, Config.MAX_TURNS)

def get_strategy_arrays_diff(timeline: list[dict]) -> tuple[list[float], list[float]]:
    """
    This function generates an array that computes
    the difference of most used Gen1 recovery moves  
    between p1 and p2
    """
    recovery_diff_raw = []
    p1_recovery_cumul = 0
    p2_recovery_cumul = 0
    
    RECOVERY_MOVES = {'recover', 'soft-boiled'}
    
    for turn in timeline:
        
        if turn is None:
            recovery_diff_raw.append(float(p1_recovery_cumul - p2_recovery_cumul))
            continue

        
        p1_move_details = turn.get('p1_move_details')
        p2_move_details = turn.get('p2_move_details')
        
        p1_move_name = (p1_move_details or {}).get('name', '').lower()
        p2_move_name = (p2_move_details or {}).get('name', '').lower()
        
        if p1_move_name in RECOVERY_MOVES: p1_recovery_cumul += 1
        
        if p2_move_name in RECOVERY_MOVES: p2_recovery_cumul += 1
            
        recovery_diff_raw.append(float(p1_recovery_cumul - p2_recovery_cumul))
        
    return pad_array(recovery_diff_raw, Config.MAX_TURNS)



def create_features(data: list[dict]) -> pd.DataFrame:
    """
    Main function to create feature
    """
    feature_list = []
    id_column = Config.ID_COLUMN_NAME
    target_column = Config.TARGET_COLUMN_NAME

    for idx, battle in enumerate(tqdm(data, desc="Extracting features")):
        features = {}
        p1_team = battle.get('p1_team_details', [])
        p2_lead = battle.get('p2_lead_details', {})
        timeline_raw = battle.get('battle_timeline', [])

        if not p1_team or not p2_lead or not timeline_raw or len(p1_team) != 6:
            print(f"Skipping battle ID {battle.get(id_column, 'Unknown')} due to missing/incomplete data.", file=sys.stderr)
            continue

        timeline = timeline_raw[:Config.MAX_TURNS]

        try:

            #Static Feature
            features.update(get_avg_stats(p1_team))
            features.update(get_avg_crit_rate(p1_team))
            features['p1_threat_score'] = get_key_threat_score(p1_team, p2_lead, timeline)
            features.update(get_team_type_scores(p1_team))
            features['starting_speed_adv'] = get_starting_superiority(p1_team, p2_lead, timeline_raw)

            #Dynamic Feature
            fainted = get_fainted_array(timeline)
            hp_diff, low_hp = get_hp_arrays(timeline, p1_team)
            status = get_status_array(timeline, p1_team) 
            boost = get_boost_array(timeline)
            switch = get_switch_array(timeline)
            p1_stab = get_p1_stab_array(timeline, p1_team)
            recovery_diff = get_strategy_arrays_diff(timeline) 


            dynamic_arrays = {
                'fainted': fainted, 'hpdiff': hp_diff, 'status': status,
                'boost': boost, 'lowhp': low_hp, 'switch': switch,
                'recovery_diff': recovery_diff 
            }
            for prefix, arr in dynamic_arrays.items():
                summary = calculate_summary_stats(arr)
                features[f'{prefix}_last'], features[f'{prefix}_avg'], features[f'{prefix}_max'], \
                features[f'{prefix}_min'], features[f'{prefix}_std'] = summary
            
            features['p1_stab'] = p1_stab[-1]
            features['p1_stab_avg']=np.mean(p1_stab)
            features['p1_stab_std_dev']=np.std(p1_stab)
            
            #Momentum features 
            features['fainted_momentum'] = calculate_momentum(fainted)
            features['hpdiff_momentum'] = calculate_momentum(hp_diff)


            features[id_column] = battle.get(id_column)
            if target_column in battle:
                features[target_column] = int(battle[target_column])

            feature_list.append(features)

        except Exception as e:
            print(f"Error processing battle ID {battle.get(id_column, 'Unknown')}: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    df = pd.DataFrame(feature_list)
    if target_column in df.columns:
        df[target_column] = df[target_column].astype('Int64')
    df = df.fillna(0.0)
    return df
