"""
Optimized prompts for Vision Agent parameter extraction
"""

DATASET_PROMPTS = {
    'synergy': """Extract EXACT numerical values for nanoparticle characterization.

FIND THESE SPECIFIC VALUES:

1. NP_size_avg_nm (average particle size in nanometers):
   - Look for: particle size, diameter, TEM size, DLS size, nanoparticle size
   - Extract: the NUMBER only (e.g., 3, 5.5, 10, 150)
   - Units must be nm (if in μm, multiply by 1000)
   - Location: tables, TEM images, DLS graphs, characterization data
   - IMPORTANT: Extract the smallest/primary size if multiple sizes shown

2. zeta_potential_mV (surface charge in millivolts):
   - Look for: zeta potential, ζ-potential, surface charge, ZP
   - Extract: the NUMBER with sign (e.g., -25.5, +10, -30)
   - Units must be mV
   - Location: zeta potential graphs, characterization tables
   - Can be positive or negative

CRITICAL INSTRUCTIONS:
- Extract EXACT numbers from the document
- For size: if you see "3-5 nm", extract 4 (middle value)
- For size: if you see "3 ± 0.5 nm", extract 3 (main value)
- Look CAREFULLY at scales on TEM images
- Check figure captions for stated values

Return JSON ARRAY (even for single measurement):
[
  {
    "NP_size_avg_nm": number or null,
    "zeta_potential_mV": number or null
  }
]
""",

    'cytox': """You are an expert in analyzing nanoparticle characterization data from cytotoxicity studies.

ROLE: Specialist in nanoparticle physical characterization
CONTEXT: Extracting size and zeta potential measurements from characterization data in cytotoxicity papers

PARAMETERS TO EXTRACT:

1. size_in_medium_nm (hydrodynamic size in cell culture medium):
   - Alternative names: hydrodynamic diameter, DLS size, particle size in medium, size in DMEM/RPMI
   - Units: nm (nanometers), convert μm to nm if needed (1 μm = 1000 nm)
   - Typical range: 50-500 nm (often larger than TEM size due to protein corona)
   - Where to look:
     * DLS (Dynamic Light Scattering) measurements in medium
     * Characterization tables with "in medium" or "in serum" data
     * Supplementary characterization data
     * Figure captions mentioning size in cell culture conditions
   - Note: Size in medium is typically larger than dry TEM size

2. zeta_in_medium_mv (zeta potential in cell culture medium):
   - Alternative names: ζ-potential in medium, surface charge in serum, zeta in DMEM/RPMI
   - Units: mV (millivolts)
   - Typical range: -30 to +30 mV (often less negative due to protein adsorption)
   - Where to look:
     * Zeta potential measurements in cell culture medium
     * Tables with "in medium" or "with serum" conditions
     * Stability characterization sections
     * Colloidal stability data
   - Note: Zeta in medium differs from zeta in water

EXTRACTION STRATEGY:
1. Look for DLS/zeta measurements specifically in cell culture medium
2. Check supplementary information for detailed characterization
3. Find tables comparing "in water" vs "in medium" measurements
4. Look for protein corona or serum effects on size/charge

VALIDATION:
- Size in medium typically 1.5-3x larger than TEM size
- Zeta in medium often less negative than in water
- Both measurements should be in biological conditions

OUTPUT FORMAT (return ARRAY even for single measurement):
[
  {
    "size_in_medium_nm": number or null,
    "zeta_in_medium_mv": number or null
  }
]
""",

    'magnetic': """Extract ALL magnetic characterization experiments from this page.

IMPORTANT: One PDF may contain MULTIPLE experiments/samples. Extract each separately!

LOOK FOR THESE SPECIFIC PARAMETERS:

1. squid_sat_mag (saturation magnetization):
   - Look for: Ms, saturation magnetization, magnetic saturation, Msat
   - Units: emu/g, Am²/kg, emu/mol, A·m²/kg
   - Location: SQUID data, VSM measurements, magnetic property tables, hysteresis loops

2. squid_temperature (measurement temperature):
   - Look for: temperature, T, measurement temp, SQUID temperature
   - Units: K (Kelvin) - THIS IS CRITICAL
   - Common values: 5K, 10K, 300K, room temperature (=300K)
   - Location: SQUID measurement conditions, figure captions, methods section

3. quid_h_max (maximum applied field):
   - Look for: Hmax, maximum field, field range, applied field
   - Units: kOe, Oe, T (Tesla), mT
   - If in Oe: divide by 1000 to get kOe
   - If in Tesla: 1T = 10 kOe
   - Location: hysteresis loop axis labels, SQUID parameters

4. np_hydro_size (hydrodynamic size):
   - Look for: hydrodynamic size, DLS size, hydrodynamic diameter
   - Units: nm (nanometers)
   - Location: DLS data, characterization tables, size distribution graphs

5. mri_r1 (longitudinal relaxivity):
   - Look for: r1, T1 relaxivity, longitudinal relaxivity, R1
   - Units: mM⁻¹s⁻¹, s⁻¹mM⁻¹, (mM·s)⁻¹
   - Location: MRI characterization, relaxivity plots, tables

6. mri_r2 (transverse relaxivity):
   - Look for: r2, T2 relaxivity, transverse relaxivity, R2
   - Units: mM⁻¹s⁻¹, s⁻¹mM⁻¹, (mM·s)⁻¹
   - Location: MRI characterization, relaxivity plots, tables

CRITICAL EXTRACTION RULES:
- For temperature: Look for "measured at X K" or "T = X K" in captions
- For Hmax: Check the x-axis range of hysteresis loops
- Extract ALL parameters you can find, not just some
- If multiple samples/experiments: return ARRAY of objects
- If single experiment: return ARRAY with one object

Return JSON ARRAY (even for single experiment):
[
  {
    "sample_id": "Sample1" or null,
    "squid_sat_mag": number or null,
    "squid_temperature": number or null,
    "quid_h_max": number or null,
    "np_hydro_size": number or null,
    "mri_r1": number or null,
    "mri_r2": number or null
  },
  {
    "sample_id": "Sample2" or null,
    "squid_sat_mag": number or null,
    "squid_temperature": number or null,
    "quid_h_max": number or null,
    "np_hydro_size": number or null,
    "mri_r1": number or null,
    "mri_r2": number or null
  }
]
""",

    'nanozymes': """CRITICAL: Nanozyme kinetics paper - Extract ALL experiments/reactions!

IMPORTANT: One PDF may test MULTIPLE substrates (TMB, ABTS, H2O2, etc). Extract EACH separately!

PART 1 - CONCENTRATION RANGE (c_min, c_max):
For any concentration vs velocity plot (v vs [S]):
1. Look at the X-axis (concentration axis)
2. Find the LEFTMOST data point → This is c_min
3. Find the RIGHTMOST data point → This is c_max
4. Units are typically: mM, μM, nM (1000 μM = 1 mM)
5. If you see multiple plots, take the first one

EXAMPLES OF CONCENTRATION RANGES:
• TMB: 0.1 to 1.3 mM → c_min: 0.1, c_max: 1.3
• H₂O₂: 10 to 130 mM → c_min: 10, c_max: 130
• ABTS: 50 to 750 μM → c_min: 50, c_max: 750
• If axis shows "Concentration (mM)" from 0 to 2 → c_min: 0, c_max: 2

PART 2 - IDENTIFY GRAPH TYPE FOR Km/Vmax:
• CURVED LINE (Michaelis-Menten): v vs [S] - direct plot
• STRAIGHT LINE (Lineweaver-Burk): 1/v vs 1/[S] - reciprocal plot
• TABLE: Direct values listed

PART 3 - EXTRACT Km AND Vmax:

FOR MICHAELIS-MENTEN (curved):
- Y-axis plateau value = Vmax
- X-axis at Y=Vmax/2 = Km
- Check caption for "Km = X mM"

FOR LINEWEAVER-BURK (straight line):
- X-intercept: Km = -1/(x-intercept)
- Y-intercept: Vmax = 1/(y-intercept)

FOR TABLES:
- Look for "Km", "KM", "Michaelis constant"
- Look for "Vmax", "V_max", "Maximum velocity"

CRITICAL EXTRACTION PRIORITY:
1. ALWAYS extract c_min and c_max from concentration axis
2. Extract Km and Vmax from plots or tables
3. Note units carefully (μM vs mM is 1000x difference!)
4. For concentration plots: actual data points, not fitted curves

REAL EXAMPLES:
• Concentration axis 0.1-1.0 mM → c_min: 0.1, c_max: 1.0
• "Km = 0.27 mM" → km_value: 0.27, km_unit: "mM"
• "Vmax = 8.611 × 10⁻⁸ M·s⁻¹" → vmax_value: 0.00000008611, vmax_unit: "M·s⁻¹"

MULTIPLE EXPERIMENTS:
If you see multiple plots (e.g., TMB, H2O2, ABTS), extract EACH as separate object!
If you see table with multiple rows, extract EACH row as separate object!

Return JSON ARRAY (even for single experiment):
[
  {
    "substrate": "TMB" or null,
    "c_min": number or null,
    "c_max": number or null,
    "km_value": number or null,
    "km_unit": "string" or null,
    "vmax_value": number or null,
    "vmax_unit": "string" or null
  },
  {
    "substrate": "H2O2" or null,
    "c_min": number or null,
    "c_max": number or null,
    "km_value": number or null,
    "km_unit": "string" or null,
    "vmax_value": number or null,
    "vmax_unit": "string" or null
  }
]
""",

    'seltox': """Extract nanoparticle characterization data from selective toxicity studies.

LOOK FOR THESE EXACT PARAMETERS:

1. np_size_avg_nm (average nanoparticle size):
   - Look for: particle size, diameter, TEM size, DLS size, average size
   - Units: nm (nanometers)
   - Location: characterization tables, TEM/SEM data, DLS measurements
   - Extract the average/mean value

2. zeta_potential_mV (surface charge):
   - Look for: zeta potential, ζ-potential, surface charge, ZP
   - Units: mV (millivolts)
   - Location: zeta potential measurements, characterization tables
   - Can be positive or negative value

IMPORTANT: 
- Extract EXACT numerical values from the document
- Look in characterization sections, tables, and figure captions
- These are PHYSICAL properties of nanoparticles, not biological effects

Return JSON ARRAY (even for single measurement):
[
  {
    "np_size_avg_nm": number or null,
    "zeta_potential_mV": number or null
  }
]
"""
}

# Parameter mappings for each dataset type (aligned with gold standard)
DATASET_PARAMETERS = {
    'synergy': ['NP_size_avg_nm', 'zeta_potential_mV'],  # Removed CI - not in gold
    'cytox': ['size_in_medium_nm', 'zeta_in_medium_mv'],  # Changed to match gold standard
    'magnetic': ['squid_sat_mag', 'squid_temperature', 'quid_h_max', 'np_hydro_size', 'mri_r1', 'mri_r2'],  # Added missing params
    'nanozymes': ['c_min', 'c_max', 'km_value', 'vmax_value', 'km_unit', 'vmax_unit'],  # Added c_min, c_max
    'seltox': ['np_size_avg_nm', 'zeta_potential_mV']  # Aligned with gold
}

# Valid ranges for parameter validation
PARAMETER_RANGES = {
    # Synergy & Seltox
    'NP_size_avg_nm': (0.1, 1000),
    'np_size_avg_nm': (0.1, 1000),
    'zeta_potential_mV': (-100, 100),

    # Cytox
    'size_in_medium_nm': (10, 2000),
    'zeta_in_medium_mv': (-100, 100),

    # Magnetic
    'squid_sat_mag': (0, 300),
    'squid_temperature': (0, 400),  # Usually 5K, 10K, 300K
    'quid_h_max': (0, 100),  # kOe
    'np_hydro_size': (1, 1000),  # nm
    'mri_r1': (0, 100),
    'mri_r2': (0, 1000),

    # Nanozymes
    'c_min': (0, 1000),  # Concentration min
    'c_max': (0.001, 10000),  # Concentration max
    'km_value': (0.0001, 1000),
    'vmax_value': (0.00000001, 10000),  # Can be very small for M·s⁻¹
    'km_unit': None,  # Text field
    'vmax_unit': None  # Text field
}

def get_prompt_for_dataset(dataset_name: str) -> str:
    """Get the optimized prompt for a specific dataset"""
    return DATASET_PROMPTS.get(dataset_name.lower(), DATASET_PROMPTS['synergy'])

def get_parameters_for_dataset(dataset_name: str) -> list:
    """Get the list of parameters to extract for a specific dataset"""
    return DATASET_PARAMETERS.get(dataset_name.lower(), [])

def validate_parameter_value(param_name: str, value: float) -> bool:
    """Validate if a parameter value is within expected range"""
    if param_name in PARAMETER_RANGES:
        min_val, max_val = PARAMETER_RANGES[param_name]
        return min_val <= value <= max_val
    return True