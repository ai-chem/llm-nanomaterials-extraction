from pydantic import BaseModel, Field
from typing import Optional

class NanozymeExperiment(BaseModel):
    formula: Optional[str] = Field(
        description="The chemical formula of the nanozyme, e.g. 'Fe3O4', 'CuO', etc. If not detected, set to 'NOT_DETECTED'."
    )
    activity: Optional[str] = Field(
        description="Catalytic activity type, typically 'peroxidase', 'oxidase', 'catalase', 'laccase', or other. If not detected, set to 'NOT_DETECTED'."
    )
    syngony: Optional[str] = Field(
        description="The crystal unit of the nanozyme, e.g. 'cubic', 'hexagonal', 'tetragonal', 'monoclinic', 'orthorhombic', 'trigonal', 'amorphous', 'triclinic'. If not detected, set to 'NOT_DETECTED'."
    )
    length: Optional[float] = Field(
        description="The length of the nanozyme particle in nanometers."
    )
    width: Optional[float] = Field(
        description="The width of the nanozyme particle in nanometers."
    )
    depth: Optional[float] = Field(
        description="The depth of the nanozyme particle in nanometers."
    )
    surface: Optional[str] = Field(
        description="The molecule on the surface of the nanozyme, e.g. 'naked', 'poly(ethylene oxide)', 'poly(N-Vinylpyrrolidone)', 'Tetrakis(4-carboxyphenyl)porphine', or other. If not detected, set to 'NOT_DETECTED'."
    )
    km_value: Optional[float] = Field(
        description="The Michaelis constant value for the nanozyme."
    )
    km_unit: Optional[str] = Field(
        description="The unit for the Michaelis constant, e.g. 'mM', etc. If not detected, set to 'NOT_DETECTED'."
    )
    vmax_value: Optional[float] = Field(
        description="The molar maximum reaction rate value."
    )
    vmax_unit: Optional[str] = Field(
        description="The unit for the maximum reaction rate, e.g. 'µmol/min', 'mol/min', etc. If not detected, set to 'NOT_DETECTED'."
    )
    reaction_type: Optional[str] = Field(
        description="The reaction type involving the substrate and co-substrate, e.g. 'TMB + H2O2', 'H2O2 + TMB', 'TMB', 'ABTS + H2O2', 'H2O2', 'OPD + H2O2', 'H2O2 + GSH', or other. If not detected, set to 'NOT_DETECTED'."
    )
    c_min: Optional[float] = Field(
        description="The minimum substrate concentration in catalytic assays in mM."
    )
    c_max: Optional[float] = Field(
        description="The maximum substrate concentration in catalytic assays in mM."
    )
    c_const: Optional[float] = Field(
        description="The constant co-substrate concentration used during assays."
    )
    c_const_unit: Optional[str] = Field(
        description="The unit of measurement for co-substrate concentration. If not detected, set to 'NOT_DETECTED'."
    )
    ccat_value: Optional[float] = Field(
        description="The concentration of the catalyst used in assays."
    )
    ccat_unit: Optional[str] = Field(
        description="The unit of measurement for catalyst concentration. If not detected, set to 'NOT_DETECTED'."
    )
    ph: Optional[float] = Field(
        description="The pH level at which experiments were conducted."
    )
    temperature: Optional[float] = Field(
        description="The temperature in Celsius during the study."
    )

class MagneticExperiment(BaseModel):
    name: Optional[str] = Field(
        description="Material name (e.g., BFO, cobalt irin oxide and bismuth ferrite etc.). If not detected, set to 'NOT_DETECTED'."
    )
    np_core: Optional[str] = Field(
        description="Composition of material core (e.g., Gd2O3, Fe1Fe2O4 etc.). If not detected, set to 'NOT_DETECTED'."
    )
    np_shell: Optional[str] = Field(
        description="Composition of material shell (e.g., chitosan, Au1 etc.). If not detected, set to 'NOT_DETECTED'."
    )
    core_shell_formula: Optional[str] = Field(
        description="Nanoparticle composition as one formula, core and shell separated by -, /, @, or | (e.g., Cr2O3-Co). If not detected, set to 'NOT_DETECTED'."
    )
    np_shell_2: Optional[str] = Field(
        description="First additional shell layer if present (e.g., PEG-5000, Curcumin etc.). If not detected, set to 'NOT_DETECTED'."
    )
    np_shell_3: Optional[str] = Field(
        description="Second additional shell layer if present (e.g., PEG-5000, Curcumin etc.). If not detected, set to 'NOT_DETECTED'."
    )
    crystal_structure_core_shell: Optional[str] = Field(
        description="Crystallographic structures of core and shell materials, comma-separated (e.g., hexagonal, cubic etc.). If not detected, set to 'NOT_DETECTED'."
    )
    space_group_core: Optional[str] = Field(
        description="Space group of core material (e.g., fd-3m, p4/mmm, etc.). If not detected, set to 'NOT_DETECTED'."
    )
    space_group_shell: Optional[str] = Field(
        description="Space group of shell material (e.g., fd-3m, p4/mmm, etc.). If not detected, set to 'NOT_DETECTED'."
    )
    xrd_crystallinity: Optional[str] = Field(
        description="Crystallinity (e.g., crystalline, amorphous, partially crystalline etc.). If not detected, set to 'NOT_DETECTED'."
    )
    instrument: Optional[str] = Field(
        description="Experimental instrument (e.g., Quantum Design 7 T SQUID magnetometer, Seifert XRD 3000P, etc.). If not detected, set to 'NOT_DETECTED'."
    )
    np_hydro_size: Optional[float] = Field(
        description="Size of nanoparticles in solution from DLS, in nm."
    )
    xrd_scherrer_size: Optional[float] = Field(
        description="Crystal size from XRD (Scherrer), in nm."
    )
    emic_size: Optional[float] = Field(
        description="Electron microscopy measured size, in nm."
    )
    squid_sat_mag: Optional[float] = Field(
        description="Saturation magnetization (Ms, Bs) in emu/g. If reported in other units, convert (1 A·m2/kg = 1 emu/g, 1 μ0M(T) = 0.01257 emu/g)."
    )
    squid_rem_mag: Optional[float] = Field(
        description="Remanent magnetization (Mr, Br) in emu/g. If reported in other units, convert (1 A·m2/kg = 1 emu/g, 1 μ0M(T) = 0.01257 emu/g)."
    )
    exchange_bias_shift_Oe: Optional[float] = Field(
        description="Exchange bias in Oersted (Oe). If reported in other units, convert (1T = 1000 Oe, 1 mT = 10 Oe, 1kOe = 1000 Oe). Do not alter the sign; if not explicitly given, assume positive (+)."
    )
    vertical_loop_shift_M_vsl_emu_g: Optional[float] = Field(
        description="Vertical loop shift in emu/g (preserve +/- sign, default to plus if not stated). If reported in other units, convert (1 A·m2/kg = 1 emu/g, 1 μ0M(T) = 0.01257 emu/g)."
    )
    hc_kOe: Optional[float] = Field(
        description="Coercivity (Hc, coercive force) in Oersted (Oe). If reported in other units, convert (1T = 1000 Oe, 1 mT = 10 Oe, 1kOe = 1000 Oe)."
    )
    squid_h_max: Optional[float] = Field(
        description="Maximum magnetic field in kOe."
    )
    zfc_h_meas: Optional[float] = Field(
        description="Measurement field for ZFC in kOe."
    )
    fc_field_T: Optional[float] = Field(
        description="FC field in Tesla (T)."
    )
    squid_temperature: Optional[float] = Field(
        description="SQUID temperature in Kelvin."
    )
    coercivity: Optional[float] = Field(
        description="Coercivity (Hc) in kOe. If reported in other units, convert to Oe."
    )
    htherm_sar: Optional[float] = Field(
        description="Specific absorption rate (SAR) in W/g."
    )
    mri_r1: Optional[float] = Field(
        description="MRI relaxation rate r1 in mM-1·s-1."
    )
    mri_r2: Optional[float] = Field(
        description="MRI relaxation rate r2 in mM-1·s-1."
    )
    blocking_temperature_K: Optional[float] = Field(
        description="Blocking temperature in Kelvin (K)."
    )
    curie_temperature_K: Optional[float] = Field(
        description="Curie temperature in Kelvin (K)."
    )

class CytotoxicityExperiment(BaseModel):
    material: Optional[str] = Field(description="Composition of the nanoparticle/material tested")
    shape: Optional[str] = Field(description="Physical shape of particle")
    coat_functional_group: Optional[str] = Field(
        default=None, description="Surface coating or functionalization"
    )
    synthesis_method: Optional[str] = Field(
        default=None, description="Synthesis method"
    )
    surface_charge: Optional[str] = Field(
        default=None, description='"Negative", "Neutral", or "Positive"'
    )
    core_nm: Optional[float] = Field(
        default=None, description="Primary particle size in nm"
    )
    size_in_medium_nm: Optional[float] = Field(
        default=None, description="Hydrodynamic size in biological medium in nm"
    )
    hydrodynamic_nm: Optional[float] = Field(
        default=None, description="Size in solution including coatings in nm"
    )
    potential_mv: Optional[float] = Field(
        default=None, description="Surface charge in solution in mV"
    )
    zeta_in_medium_mv: Optional[float] = Field(
        default=None, description="Zeta potential in medium in mV"
    )
    no_of_cells_cells_well: Optional[int] = Field(
        default=None, description="Cell density per well in the assay"
    )
    human_animal: Optional[str] = Field(
        description='"A" or "H"; "A" = Animal, "H" = Human'
    )
    cell_source: Optional[str] = Field(
        description="Species/organism"
    )
    cell_tissue: Optional[str] = Field(
        description="Tissue origin of the cell line"
    )
    cell_morphology: Optional[str] = Field(
        default=None, description="Cell shape"
    )
    cell_age: Optional[str] = Field(
        default=None, description="Developmental stage of cells"
    )
    time_hr: Optional[float] = Field(
        description="Exposure duration in hours"
    )
    concentration: Optional[float] = Field(
        description="Tested concentration of the material"
    )
    test: Optional[str] = Field(
        description="Cytotoxicity assay type"
    )
    test_indicator: Optional[str] = Field(
        description="Reagent measured"
    )
    viability_: Optional[float] = Field(
        alias="viability_%",
        description="Cell viability percentage relative to control"
    )

class SeltoxExperiment(BaseModel):
    np: Optional[str] = Field(description='Nanoparticle name (e.g., "Ag", "Au", "ZnO")')
    coating: Optional[str] = Field(description='Surface coating/modification ("1" for coating, "0" for none)')
    bacteria: Optional[str] = Field(description='Bacterial strain tested (e.g., "Escherichia coli", "Staphylococcus aureus")')
    mdr: Optional[int] = Field(description='Multidrug-resistant strain indicator, 1 or 0 (1 for MDR, 0 for not MDR)')
    strain: Optional[str] = Field(description='Specific strain identifier (e.g., "ATCC 25922")')
    np_synthesis: Optional[str] = Field(description='Synthesis method (e.g., "green_synthesis", "chemical_synthesis", or detailed)')
    method: Optional[str] = Field(description='Assay type (e.g., "MIC", "ZOI", "MBC", "MBEC")')
    mic_np_µg_ml: Optional[float] = Field(description='MIC in μg/mL')
    concentration: Optional[float] = Field(description='Concentration for ZOI in μg/mL')
    zoi_np_mm: Optional[float] = Field(description='Zone of Inhibition in mm')
    np_size_min_nm: Optional[float] = Field(description='Minimum nanoparticle size in nm')
    np_size_max_nm: Optional[float] = Field(description='Maximum nanoparticle size in nm')
    np_size_avg_nm: Optional[float] = Field(description='Average nanoparticle size in nm')
    shape: Optional[str] = Field(description='Nanoparticle morphology (e.g., "spherical", "triangular")')
    time_set_hours: Optional[float] = Field(description='Experiment duration in hours')
    zeta_potential_mV: Optional[float] = Field(description='Surface charge in mV')
    solvent_for_extract: Optional[str] = Field(description='Solvent in green synthesis (e.g., "water", "ethanol")')
    temperature_for_extract_C: Optional[float] = Field(description='Extract preparation temp in °C')
    duration_preparing_extract_min: Optional[float] = Field(description='Extract prep time in minutes')
    precursor_of_np: Optional[str] = Field(description='Chemical precursor (e.g., "AgNO3")')
    concentration_of_precursor_mM: Optional[float] = Field(description='Precursor concentration in mM')
    hydrodynamic_diameter_nm: Optional[float] = Field(description='Hydrodynamic size in nm')
    ph_during_synthesis: Optional[float] = Field(description='pH of synthesis solution')

class SynergyExperiment(BaseModel):
    NP: Optional[str] = Field(description='Nanoparticle name as cited in the text, e.g. "Ag", "Au"')
    bacteria: Optional[str] = Field(description='Bacterial strain tested, e.g. "Escherichia coli"')
    strain: Optional[str] = Field(description='Specific strain identifier for the bacteria tested as cited in the text, e.g. "ATCC 25922", "MTCC 443"')
    NP_synthesis: Optional[str] = Field(description='Method by which the nanoparticles were synthesized, e.g. "chemical synthesis", "hydrothermal synthesis"')
    drug: Optional[str] = Field(description='Name of the conventional antibiotic or other antimicrobial drug used in combination with the nanoparticles, e.g. "Ampicillin"')
    drug_dose_µg_disk: Optional[float] = Field(description='Specific dosage or concentration of the drug applied, primarily used for methods like disc diffusion assays, typically measured in micrograms per disk')
    NP_concentration_µg_ml: Optional[float] = Field(description='Concentration of the nanoparticle used in the antibacterial assay, typically measured in micrograms per milliliter')
    NP_size_min_nm: Optional[float] = Field(description='Smallest recorded size of the nanoparticle, measured in nanometers')
    NP_size_max_nm: Optional[float] = Field(description='Largest recorded size of the nanoparticle, measured in nanometers')
    NP_size_avg_nm: Optional[float] = Field(description='Average nanoparticle size, typically based on TEM or DLS, measured in nanometers')
    shape: Optional[str] = Field(description='Observed morphology or physical shape of the nanoparticles, e.g. "spherical", "rod-shaped"')
    method: Optional[str] = Field(description='Experimental technique employed to assess the antibacterial efficacy, e.g. "MIC", "disc_diffusion"')
    ZOI_drug_mm_or_MIC__µg_m: Optional[float] = Field(description='Quantitative measure of antibacterial activity for the drug alone: ZOI diameter in mm or MIC in µg/mL')
    error_ZOI_drug_mm_or_MIC_µg_ml: Optional[float] = Field(description='Uncertainty/variability for the antibacterial activity measurement for the drug alone')
    ZOI_NP_mm_or_MIC_np_µg_ml: Optional[float] = Field(description='Quantitative measure of antibacterial activity for the nanoparticle alone: ZOI diameter in mm or MIC in µg/mL')
    error_ZOI_NP_mm_or_MIC_np_µg_ml: Optional[float] = Field(description='Uncertainty/variability for the antibacterial activity measurement for the nanoparticle alone')
    ZOI_drug_NP_mm_or_MIC_drug_NP_µg_ml: Optional[float] = Field(description='Quantitative measure of antibacterial activity for the combination of the drug and nanoparticle')
    error_ZOI_drug_NP_mm_or_MIC_drug_NP_µg_ml: Optional[float] = Field(description='Uncertainty/variability for the antibacterial activity measurement for the combination')
    fold_increase_in_antibacterial_activity: Optional[float] = Field(description='How much more effective the combination is compared to the most effective single component')
    zeta_potential_mV: Optional[float] = Field(description='Electrokinetic potential of the nanoparticle surface, measured in mV')
    MDR: Optional[str] = Field(description='Indicator of whether the bacterial strain is multidrug resistant, e.g. "Yes", "No", "Resistant", "Susceptible"')
    FIC: Optional[float] = Field(description='Fractional Inhibitory Concentration index value')
    effect: Optional[str] = Field(description='Qualitative description of the interaction between drug and nanoparticle based on the FIC index')
    time_hr: Optional[float] = Field(description='Duration of exposure of the bacterial cells to the antibacterial agents in hours')
    coating_with_antimicrobial_peptide_polymers: Optional[str] = Field(description='Whether the nanoparticles were modified with antimicrobial peptides or polymers, e.g. "yes", "no", material name')
    combined_MIC: Optional[float] = Field(description='Minimum Inhibitory Concentration for the combination in µg/mL')
    peptide_MIC: Optional[float] = Field(description='Minimum Inhibitory Concentration of the antimicrobial peptide alone, in µg/mL')
    viability_: Optional[float] = Field(description='Percentage of surviving/viable bacterial cells after exposure')
    viability_error: Optional[float] = Field(description='Associated error or standard deviation for the bacterial viability percentage measurement')