"""
Dataset-specific prompts and parameter extraction configurations
Each dataset has its own set of parameters to extract from graphs and tables
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DatasetType(Enum):
    """Types of datasets with specific parameters"""
    NANOZYMES = "nanozymes"
    CYTOTOXICITY = "cytotoxicity"
    MAGNETIC = "magnetic"
    SELECTIVE_TOXICITY = "selective_toxicity"
    SYNERGY = "synergy"


# ============= NANOZYMES Dataset Models =============

class NanozymesConcentrationData(BaseModel):
    """Concentration data for nanozymes dataset"""
    reaction_type: str = Field(description="Type of reaction (e.g. TMB+H2O2, ABTS+H2O2)")
    c_min: float = Field(description="Minimum concentration value")
    c_max: float = Field(description="Maximum concentration value")
    c_unit: str = Field(default="mM", description="Concentration unit (mM, µM, nM)")
    co_substrate_concentration: Optional[float] = Field(None, description="Co-substrate concentration if specified")


class NanozymesKineticParameters(BaseModel):
    """Kinetic parameters for nanozymes"""
    km_value: Optional[float] = Field(None, description="Michaelis constant Km value")
    km_unit: Optional[str] = Field(None, description="Km unit (mM, µM)")
    vmax_value: Optional[float] = Field(None, description="Maximum reaction rate Vmax value")
    vmax_unit: Optional[str] = Field(None, description="Vmax unit (mM/s, Ms-1)")
    kcat: Optional[float] = Field(None, description="Turnover number kcat in s^-1")


class NanozymesAnalysis(BaseModel):
    """Complete analysis for nanozymes dataset"""
    figure_id: Optional[str] = Field(None, description="Figure identifier (e.g., fig s15, fig 4)")
    concentration_data: Optional[List[NanozymesConcentrationData]] = None
    kinetic_parameters: Optional[NanozymesKineticParameters] = None
    description: str = Field(description="Description of what was found")


# ============= CYTOTOXICITY Dataset Models =============

class CytotoxicityData(BaseModel):
    """Data model for cytotoxicity measurements"""
    figure_id: Optional[str] = Field(None, description="Figure identifier")
    size_in_medium_nm: Optional[float] = Field(None, description="Nanoparticle size in medium (nm)")
    zeta_in_medium_mv: Optional[float] = Field(None, description="Zeta potential in medium (mV)")
    cell_viability_percent: Optional[List[float]] = Field(None, description="Cell viability percentages at different concentrations")
    ic50_value: Optional[float] = Field(None, description="IC50 value if present")
    ic50_unit: Optional[str] = Field(None, description="IC50 unit (µg/mL, nM, etc.)")
    concentration_range_min: Optional[float] = Field(None, description="Minimum concentration tested")
    concentration_range_max: Optional[float] = Field(None, description="Maximum concentration tested")
    concentration_unit: Optional[str] = Field(None, description="Concentration unit")


# ============= MAGNETIC Dataset Models =============

class MagneticData(BaseModel):
    """Data model for magnetic properties"""
    figure_id: Optional[str] = Field(None, description="Figure identifier")
    squid_h_max: Optional[float] = Field(None, description="Maximum magnetic field (kOe)")
    squid_temperature: Optional[float] = Field(None, description="SQUID measurement temperature (K)")
    hc_kOe: Optional[float] = Field(None, description="Coercivity (kOe)")
    ms_emu_g: Optional[float] = Field(None, description="Saturation magnetization (emu/g)")
    exchange_bias_shift_Oe: Optional[float] = Field(None, description="Exchange bias shift (Oe)")
    mri_r1: Optional[float] = Field(None, description="MRI relaxivity r1 (mM^-1 s^-1)")
    mri_r2: Optional[float] = Field(None, description="MRI relaxivity r2 (mM^-1 s^-1)")
    blocking_temperature_K: Optional[float] = Field(None, description="Blocking temperature (K)")


# ============= SELECTIVE TOXICITY Dataset Models =============

class SelectiveToxicityData(BaseModel):
    """Data model for selective toxicity measurements"""
    figure_id: Optional[str] = Field(None, description="Figure identifier")
    np_size_avg_nm: Optional[float] = Field(None, description="Average nanoparticle size (nm)")
    zeta_potential_mV: Optional[float] = Field(None, description="Zeta potential (mV)")
    cancer_cell_viability: Optional[List[float]] = Field(None, description="Cancer cell viability at different concentrations")
    normal_cell_viability: Optional[List[float]] = Field(None, description="Normal cell viability at different concentrations")
    selectivity_index: Optional[float] = Field(None, description="Selectivity index (IC50_normal/IC50_cancer)")
    ic50_cancer: Optional[float] = Field(None, description="IC50 for cancer cells")
    ic50_normal: Optional[float] = Field(None, description="IC50 for normal cells")
    concentration_unit: Optional[str] = Field(None, description="Concentration unit")


# ============= SYNERGY Dataset Models =============

class SynergyData(BaseModel):
    """Data model for synergy measurements"""
    figure_id: Optional[str] = Field(None, description="Figure identifier")
    NP_size_avg_nm: Optional[float] = Field(None, description="Average nanoparticle size (nm)")
    zeta_potential_mV: Optional[float] = Field(None, description="Zeta potential (mV)")
    combination_index: Optional[float] = Field(None, description="Combination index (CI)")
    drug_reduction_index: Optional[float] = Field(None, description="Drug reduction index")
    synergy_score: Optional[float] = Field(None, description="Synergy score")
    combined_effect: Optional[List[float]] = Field(None, description="Combined treatment effect at different concentrations")
    individual_effect_drug1: Optional[List[float]] = Field(None, description="Individual drug 1 effect")
    individual_effect_drug2: Optional[List[float]] = Field(None, description="Individual drug 2 effect")


# ============= Dataset-specific Prompts =============

def get_dataset_prompt(dataset_type: DatasetType, include_example: bool = True) -> str:
    """Get specialized prompt for each dataset type"""
    
    if dataset_type == DatasetType.NANOZYMES:
        return """
Analyze this image for NANOZYME kinetic data. Extract:

1. Concentration vs Velocity graphs:
   - X-axis: substrate concentration (look for units: mM, µM, nM)
   - Y-axis: reaction velocity/rate
   - Find c_min and c_max (concentration range)
   - Identify substrate type (TMB, ABTS, H2O2, etc.)

2. Kinetic parameters from tables or text:
   - Km value and unit (mM, µM)
   - Vmax value and unit (mM/s, Ms-1, etc.)
   - kcat (turnover number) if present

3. Figure identification (fig 1, fig s15, etc.)

IMPORTANT: Only analyze Michaelis-Menten type concentration-velocity plots.
Ignore Lineweaver-Burk plots (1/v vs 1/[S]) and other reciprocal plots.

Return as structured JSON with all found parameters.
"""

    elif dataset_type == DatasetType.CYTOTOXICITY:
        return """
Analyze this image for CYTOTOXICITY data. Extract:

1. From dose-response curves:
   - X-axis: concentration (µg/mL, nM, µM, mg/mL)
   - Y-axis: cell viability (%)
   - IC50 value if shown
   - Concentration range (min and max)

2. From characterization data:
   - Size in medium (nm) - hydrodynamic diameter
   - Zeta potential in medium (mV)

3. Cell viability values at different concentrations

4. Figure identification

Focus on graphs showing cell viability vs concentration relationships.
Extract actual data points if visible.

Return as structured JSON with all found parameters.
"""

    elif dataset_type == DatasetType.MAGNETIC:
        return """
Analyze this image for MAGNETIC properties data. Extract:

1. From hysteresis loops (M-H curves):
   - Maximum field H_max (kOe, Oe)
   - Coercivity Hc (kOe, Oe)
   - Saturation magnetization Ms (emu/g, Am²/kg)
   - Temperature of measurement (K)

2. From temperature-dependent measurements:
   - SQUID temperature (K)
   - Blocking temperature if shown
   - Exchange bias shift (Oe)

3. From MRI relaxivity plots:
   - r1 value (mM⁻¹s⁻¹)
   - r2 value (mM⁻¹s⁻¹)

4. Figure identification

Focus on M-H loops, M-T curves, ZFC-FC curves, and relaxivity plots.

Return as structured JSON with all found parameters.
"""

    elif dataset_type == DatasetType.SELECTIVE_TOXICITY:
        return """
Analyze this image for SELECTIVE TOXICITY data. Extract:

1. Nanoparticle characterization:
   - Average size (nm)
   - Zeta potential (mV)

2. From viability curves comparing cancer vs normal cells:
   - Cancer cell viability at different concentrations
   - Normal cell viability at different concentrations
   - IC50 for cancer cells
   - IC50 for normal cells
   - Selectivity index (IC50_normal/IC50_cancer)

3. Concentration units and ranges

4. Figure identification

Focus on comparative viability plots showing differential toxicity.
Look for graphs with two curves - one for cancer cells, one for normal cells.

Return as structured JSON with all found parameters.
"""

    elif dataset_type == DatasetType.SYNERGY:
        return """
Analyze this image for SYNERGY/COMBINATION therapy data. Extract:

1. Nanoparticle properties:
   - Average size (nm)
   - Zeta potential (mV)

2. From synergy analysis plots:
   - Combination index (CI) values
   - Drug reduction index
   - Synergy scores (if using specific scoring methods)

3. From dose-response curves:
   - Combined treatment effects
   - Individual drug/treatment effects
   - Concentration ranges

4. From isobolograms or heat maps:
   - Synergistic regions (CI < 1)
   - Antagonistic regions (CI > 1)

5. Figure identification

Focus on combination index plots, isobolograms, Bliss synergy maps.
CI < 1 indicates synergy, CI = 1 additive, CI > 1 antagonism.

Return as structured JSON with all found parameters.
"""
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_model(dataset_type: DatasetType):
    """Get the appropriate Pydantic model for the dataset type"""
    models = {
        DatasetType.NANOZYMES: NanozymesAnalysis,
        DatasetType.CYTOTOXICITY: CytotoxicityData,
        DatasetType.MAGNETIC: MagneticData,
        DatasetType.SELECTIVE_TOXICITY: SelectiveToxicityData,
        DatasetType.SYNERGY: SynergyData
    }
    return models.get(dataset_type)


def map_folder_to_dataset_type(folder_name: str) -> DatasetType:
    """Map folder names to dataset types"""
    mapping = {
        'cytox': DatasetType.CYTOTOXICITY,
        'cytotoxicity': DatasetType.CYTOTOXICITY,
        'magnetic': DatasetType.MAGNETIC,
        'nanozymes': DatasetType.NANOZYMES,
        'seltox': DatasetType.SELECTIVE_TOXICITY,
        'selective_toxicity': DatasetType.SELECTIVE_TOXICITY,
        'synergy': DatasetType.SYNERGY
    }
    
    folder_lower = folder_name.lower()
    for key, value in mapping.items():
        if key in folder_lower:
            return value
    
    # Default to nanozymes if not found
    return DatasetType.NANOZYMES


# Example usage function
def get_extraction_config(dataset_name: str) -> Dict:
    """Get complete extraction configuration for a dataset"""
    dataset_type = map_folder_to_dataset_type(dataset_name)
    
    return {
        'type': dataset_type,
        'prompt': get_dataset_prompt(dataset_type),
        'model': get_dataset_model(dataset_type),
        'parameters': {
            'nanozymes': ['c_min', 'c_max', 'km_value', 'km_unit', 'vmax_value', 'vmax_unit'],
            'cytotoxicity': ['size_in_medium_nm', 'zeta_in_medium_mv', 'ic50_value'],
            'magnetic': ['squid_temperature', 'hc_kOe', 'ms_emu_g', 'mri_r1', 'mri_r2'],
            'selective_toxicity': ['np_size_avg_nm', 'zeta_potential_mV', 'selectivity_index'],
            'synergy': ['NP_size_avg_nm', 'zeta_potential_mV', 'combination_index']
        }.get(dataset_type.value, [])
    }


if __name__ == "__main__":
    # Test the configuration
    for dataset in ['cytox', 'magnetic', 'nanozymes', 'seltox', 'synergy']:
        config = get_extraction_config(dataset)
        print(f"\n{dataset.upper()}:")
        print(f"  Type: {config['type'].value}")
        print(f"  Model: {config['model'].__name__}")
        print(f"  Parameters: {config['parameters']}")