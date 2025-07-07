import os
import sys
import pandas as pd
from itertools import permutations as pt
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from typing import List, Union, Literal
from ranking_phase_fields.logger import get_logger

logger = get_logger(__name__)

ELEMENT_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
    'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
    'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Fl', 'Lv'
]

class Config(BaseModel):
    icsd_file: str = "icsd2017"
    phase_fields: Literal["ternary", "quaternary", "quinary"] = "quaternary"
    cations_train: Union[str, List[str]] = "all"
    anions_train: Union[str, List[str]] = Field(default_factory=list)
    nanions_train: int = 0
    cation1_test: Union[str, List[str]] = Field(default_factory=list)
    cation2_test: Union[str, List[str]] = Field(default_factory=list)
    anion1_test: Union[str, List[str]] = Field(default_factory=list)
    anion2_test: Union[str, List[str]] = Field(default_factory=list)
    method: Literal["VAE","AE", "VAE_encoder"] = "VAE"
    cross_validate: bool = False
    average_runs: int = 1
    features: str = "magpie"
    elements_test: List[str] = Field(default_factory=list)

def parse_input(inputfile: str = "config.yaml") -> Config:
    logger.info(f"Reading input file {inputfile}")
    with open(inputfile, "r") as f:
        data = yaml.safe_load(f)

    # Convert possible comma-separated string to list
    def to_list(value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",")]
        return value

    data["anions_train"] = to_list(data.get("anions_train", []))
    data["cation1_test"] = to_list(data.get("cation1_test", []))
    data["cation2_test"] = to_list(data.get("cation2_test", []))
    data["anion1_test"] = to_list(data.get("anion1_test", []))
    data["anion2_test"] = to_list(data.get("anion2_test", []))

    # Handle "all" logic for cations_train
    if data.get("cations_train") == "all":
        data["cations_train"] = ELEMENT_SYMBOLS

    cfg = Config(**data)

    # Compute elements_test
    if cfg.phase_fields == "binary":
        cfg.elements_test = [cfg.cation1_test,cfg.anion1_test]
    elif cfg.phase_fields == "ternary" and cfg.nanions_train in [0, 2]:
        cfg.elements_test = [cfg.cation1_test, cfg.anion1_test, cfg.anion2_test]
    elif cfg.phase_fields == "ternary" and cfg.nanions_train == 1:
        cfg.elements_test = [cfg.cation1_test, cfg.cation2_test, cfg.anion1_test]
    elif cfg.phase_fields =='quaternary':
        cfg.elements_test = [cfg.cation1_test, cfg.cation2_test, cfg.anion1_test, cfg.anion2_test]
    else:
        logger.warning(f"{cfg.phase_fields} unsuported phase fields")

    logger.info("Parsed configuration:")
    for k, v in cfg.model_dump().items():
        logger.info(f"{k:15}: {v}")

    return cfg 

def numatoms(phase_fields):
    nums = {'binary': 2, 'ternary': 3, 'quaternary': 4, 'quinary': 5}
    return nums[phase_fields]

def justify(field, maxatom):
    """Zero-pad field by appending X to match maxatom size."""
    diff = maxatom - len(field)
    return field + ['X'] * diff if diff else field

def parse_icsd(phase_fields, anions_train, nanions_train, cations_train, icsd_file='icsd2017'):
    logger.info("="*55)

    if cations_train == 'all':
        cations_train = ELEMENT_SYMBOLS
    if anions_train == 'all':
        anions_train = ELEMENT_SYMBOLS

    anions_with_charge = [a + '-' for a in anions_train]

    icsd_path = Path("data") / icsd_file
    logger.info(f"Reading training data: {icsd_path}")

    # If custom CSV file
    if not ("icsd" in icsd_file.lower()):
        logger.info(f"Custom training data in {icsd_file} will be treated as list of phase fields")
        df = pd.read_csv(icsd_path)

        df['fields'] = df.iloc[:, 0].apply(lambda field: sorted(field.split()))
        logger.info(f"Original size: {len(df)}")

        # Only keep phase fields with all cations in training set
        df['bad_cations'] = df['fields'].apply(lambda f: any(el not in cations_train for el in f))
        df = df[~df['bad_cations']]
        logger.info(f"Filtered size (valid cations): {len(df)}")

        # Justify field lengths
        maxatom = df['fields'].apply(len).max()
        df['fields'] = df['fields'].apply(lambda f: justify(f, maxatom))

        fields = df['fields'].tolist()

    else:
        logger.info(f"Parsing ICSD-type file for {phase_fields} phase fields with {nanions_train} anions...")
        lines = icsd_path.read_text().splitlines()
        logger.info(f'icsd lines: {len(lines)}, {lines[0]}')
        fields = []
        n_atoms = numatoms(phase_fields)
        nanions = int(nanions_train)
        logger.info(f'nanions, {nanions}')

    for line in lines:
        tokens = line.split()
        if len(tokens) != n_atoms + 1:
            continue
     
        field, oxi = [], []
     
        for el in tokens[1:]:
            # Parse symbol
            if el[-1] in ('+', '-') and el[-2].isdigit():
                sym = el[:-2]
            elif el[-1] in ('+', '-'):
                sym = el[:-1]
            else:
                # Unexpected format
                continue
     
            ox = el
     
            if sym not in cations_train + anions_train:
                break
     
            field.append(sym)
            oxi.append(ox)
     
        else:
            if nanions > 0:
                # Count how many oxi entries are for anions
                n_anions_found = sum(1 for o in oxi if any(o.startswith(a) and o[-1] == '-' for a in anions_train))
                if n_anions_found >= nanions and sorted(field) not in fields:
                    fields.append(sorted(field))
            else:
                if sorted(field) not in fields:
                    fields.append(sorted(field))

    logger.info(f'Collected fields: {len(fields)}')
    return fields
